# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""

import math
import random
from typing import Iterable

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from tqdm.auto import tqdm

import rfdetr.util.misc as utils
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.util.logger import get_logger
from rfdetr.util.misc import get_world_size

try:
    from torch.amp import GradScaler, autocast

    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    DEPRECATED_AMP = True
from typing import Any, Callable, DefaultDict, List

import numpy as np

from rfdetr.util.misc import NestedTensor

logger = get_logger()
BYTES_TO_MB = 1024.0 * 1024.0


def _is_cuda(device: torch.device) -> bool:
    """Return True if device is a CUDA device with an active CUDA context."""
    return (
        isinstance(device, torch.device)
        and device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_initialized()
    )


def _get_autocast_dtype(device: torch.device) -> torch.dtype:
    """Return the autocast dtype appropriate for *device*.

    For non-CUDA devices the value is not used (autocast is disabled), so
    returning ``bfloat16`` is a safe no-op default.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return torch.bfloat16

    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported):
        return torch.bfloat16 if is_bf16_supported() else torch.float16

    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _get_cuda_autocast_dtype() -> torch.dtype:
    """Backward-compatible alias for CUDA autocast dtype selection."""
    return _get_autocast_dtype(torch.device("cuda"))


def get_autocast_args(args, device: torch.device):
    autocast_enabled = bool(getattr(args, "amp", False)) and device.type == "cuda"
    autocast_dtype = _get_autocast_dtype(device)
    if DEPRECATED_AMP:
        return {"enabled": autocast_enabled, "dtype": autocast_dtype}
    else:
        return {"device_type": device.type, "enabled": autocast_enabled, "dtype": autocast_dtype}


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    print_freq = args.print_freq if args is not None else 10
    start_steps = epoch * num_training_steps_per_epoch

    # Add gradient scaler for AMP
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler(device.type, enabled=args.amp)

    optimizer.zero_grad()

    # Check if batch size is divisible by gradient accumulation steps
    if batch_size % args.grad_accum_steps != 0:
        logger.error(
            f"Batch size ({batch_size}) must be divisible by gradient accumulation steps ({args.grad_accum_steps})"
        )
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by gradient accumulation steps ({args.grad_accum_steps})"
        )

    logger.info(
        f"Training config: grad_accum_steps={args.grad_accum_steps}, "
        f"total_batch_size={batch_size * get_world_size()}, "
        f"dataloader_length={len(data_loader)}"
    )

    sub_batch_size = batch_size // args.grad_accum_steps

    header = f"Epoch: [{epoch + 1}/{args.epochs}]"
    use_progress_bar = bool(getattr(args, "progress_bar", False))
    if use_progress_bar:
        progress_iter = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=header,
            colour="green",
            disable=not utils.is_main_process(),
        )
    else:
        progress_iter = enumerate(metric_logger.log_every(data_loader, print_freq, header))

    for data_iter_step, (samples, targets) in progress_iter:
        it = start_steps + data_iter_step
        callback_dict = {
            "step": it,
            "model": model,
            "epoch": epoch,
        }
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)
        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        if args.multi_scale and not args.do_random_resize_via_padding:
            scales = compute_multi_scale_scales(
                args.resolution, args.expanded_scales, args.patch_size, args.num_windows
            )
            random.seed(it)
            scale = random.choice(scales)
            with torch.no_grad():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode="bilinear", align_corners=False)
                samples.mask = (
                    F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode="nearest").squeeze(1).bool()
                )

        for i in range(args.grad_accum_steps):
            start_idx = i * sub_batch_size
            final_idx = start_idx + sub_batch_size
            new_samples_tensors = samples.tensors[start_idx:final_idx]
            new_samples = NestedTensor(new_samples_tensors, samples.mask[start_idx:final_idx])
            new_samples = new_samples.to(device)
            new_targets = [{k: v.to(device) for k, v in t.items()} for t in targets[start_idx:final_idx]]

            with autocast(**get_autocast_args(args, device)):
                outputs = model(new_samples, new_targets)
                loss_dict = criterion(outputs, new_targets)
                weight_dict = criterion.weight_dict
                losses = sum(
                    (1 / args.grad_accum_steps) * loss_dict[k] * weight_dict[k]
                    for k in loss_dict.keys()
                    if k in weight_dict
                )
                del outputs

            scaler.scale(losses).backward()

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            logger.error(f"Loss is {loss_value}, stopping training. Loss dict: {loss_dict_reduced}")
            raise ValueError(f"Loss is {loss_value}, stopping training")

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if use_progress_bar:
            log_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            postfix = {
                "lr": f"{log_dict['lr']:.6f}",
                "class_loss": f"{log_dict['class_error']:.2f}",
                "box_loss": f"{log_dict['loss_bbox']:.2f}",
                "loss": f"{log_dict['loss']:.2f}",
            }
            if _is_cuda(device):
                postfix["max_mem"] = f"{torch.cuda.max_memory_allocated(device=device) / BYTES_TO_MB:.0f} MB"
            progress_iter.set_postfix(postfix)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Epoch {epoch + 1} stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt):
    """Sweep confidence thresholds and compute precision/recall/F1 at each."""
    num_classes = len(per_class_data)
    results = []

    for conf_thresh in conf_thresholds:
        per_class_precisions = []
        per_class_recalls = []
        per_class_f1s = []

        for k in range(num_classes):
            data = per_class_data[k]
            scores = data["scores"]
            matches = data["matches"]
            ignore = data["ignore"]
            total_gt = data["total_gt"]

            above_thresh = scores >= conf_thresh
            valid = above_thresh & ~ignore

            valid_matches = matches[valid]

            tp = np.sum(valid_matches != 0)
            fp = np.sum(valid_matches == 0)
            fn = total_gt - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class_precisions.append(precision)
            per_class_recalls.append(recall)
            per_class_f1s.append(f1)

        if len(classes_with_gt) > 0:
            macro_precision = np.mean([per_class_precisions[k] for k in classes_with_gt])
            macro_recall = np.mean([per_class_recalls[k] for k in classes_with_gt])
            macro_f1 = np.mean([per_class_f1s[k] for k in classes_with_gt])
        else:
            macro_precision = 0.0
            macro_recall = 0.0
            macro_f1 = 0.0

        results.append(
            {
                "confidence_threshold": conf_thresh,
                "macro_f1": macro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "per_class_prec": np.array(per_class_precisions),
                "per_class_rec": np.array(per_class_recalls),
                "per_class_f1": np.array(per_class_f1s),
            }
        )

    return results


def _matching_data_from_coco_eval(coco_eval: Any) -> dict[int, dict[str, Any]]:
    """Bridge: extract per-class matching data from a COCOeval object.

    Converts the flat ``coco_eval.evalImgs`` list into the compact
    ``{class_id: {scores, matches, ignore, total_gt}}`` format produced by
    :func:`build_matching_data`, making legacy evaluator output compatible with
    :func:`sweep_confidence_thresholds` and :func:`merge_matching_data`.

    Only the "all" area range and IoU threshold 0.50 are used, matching the
    convention in :func:`coco_extended_metrics`.

    Args:
        coco_eval: A :class:`pycocotools.cocoeval.COCOeval` instance after
            ``evaluate()`` and ``accumulate()`` have been called.

    Returns:
        Mapping from integer category ID to per-class matching data dict with
        keys ``"scores"``, ``"matches"``, ``"ignore"`` (numpy arrays) and
        ``"total_gt"`` (int).
    """
    iou50_idx = int(np.argmax(np.isclose(coco_eval.params.iouThrs, 0.50)))
    cat_ids: list[int] = coco_eval.params.catIds
    area_rng_all = tuple(coco_eval.params.areaRng[0])

    # Unflatten flat evalImgs list → {cat_id: {area_rng: {img_id: entry}}}
    evalImgs_unflat: dict[int, dict] = {}
    for e in coco_eval.evalImgs:
        if e is None:
            continue
        cid = e["category_id"]
        evalImgs_unflat.setdefault(cid, {}).setdefault(tuple(e["aRng"]), {})[e["image_id"]] = e

    result: dict[int, dict[str, Any]] = {}
    for cid in cat_ids:
        dt_scores: list[float] = []
        dt_matches: list[float] = []
        dt_ignore: list[bool] = []
        total_gt = 0

        for img_id in coco_eval.params.imgIds:
            e = evalImgs_unflat.get(cid, {}).get(area_rng_all, {}).get(img_id)
            if e is None:
                continue
            total_gt += sum(1 for ig in e["gtIgnore"] if not ig)
            for d in range(len(e["dtIds"])):
                dt_scores.append(e["dtScores"][d])
                dt_matches.append(e["dtMatches"][iou50_idx, d])
                dt_ignore.append(bool(e["dtIgnore"][iou50_idx, d]))

        result[cid] = {
            "scores": np.array(dt_scores, dtype=float),
            "matches": np.array(dt_matches),
            "ignore": np.array(dt_ignore, dtype=bool),
            "total_gt": total_gt,
        }
    return result


def coco_extended_metrics(coco_eval: Any) -> dict[str, Any]:
    """Compute precision/recall by sweeping confidence thresholds to maximise macro-F1.

    Compatibility wrapper kept for the migration period.  Internally delegates
    to :func:`_matching_data_from_coco_eval` and
    :func:`sweep_confidence_thresholds`; does not access
    ``COCOeval.evalImgs`` directly.

    Args:
        coco_eval: A :class:`pycocotools.cocoeval.COCOeval` instance after
            ``evaluate()`` and ``accumulate()`` have been called.

    Returns:
        Dict with keys ``"class_map"`` (list of per-class metric dicts),
        ``"map"`` (mAP@50), ``"precision"``, ``"recall"``, ``"f1_score"``.
    """
    cat_ids: list[int] = coco_eval.params.catIds
    num_classes = len(cat_ids)
    area_idx = 0
    maxdet_idx = 2
    iou50_idx = int(np.argmax(np.isclose(coco_eval.params.iouThrs, 0.50)))

    class_matching = _matching_data_from_coco_eval(coco_eval)
    per_class_data = [class_matching[cid] for cid in cat_ids]

    conf_thresholds = np.linspace(0.0, 1.0, 101)
    classes_with_gt = [k for k in range(num_classes) if per_class_data[k]["total_gt"] > 0]

    confidence_sweep_metric_dicts = sweep_confidence_thresholds(per_class_data, conf_thresholds, classes_with_gt)

    best = max(confidence_sweep_metric_dicts, key=lambda x: x["macro_f1"])

    map_50_95, map_50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    per_class = []
    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    for k, cid in enumerate(cat_ids):
        # [T, R, K, A, M] -> [T, R]
        p_slice = coco_eval.eval["precision"][:, :, k, area_idx, maxdet_idx]

        # [T, R]
        p_masked = np.where(p_slice > -1, p_slice, np.nan)

        # Two sequential nanmeans to avoid underweighting columns with more nans,
        # since each column corresponds to a different IoU threshold
        # [T, R] -> [T]
        ap_per_iou = np.nanmean(p_masked, axis=1)

        # [T] -> [1]
        ap_50_95 = float(np.nanmean(ap_per_iou))
        ap_50 = float(np.nanmean(p_masked[iou50_idx]))

        if (
            np.isnan(ap_50_95)
            or np.isnan(ap_50)
            or np.isnan(best["per_class_prec"][k])
            or np.isnan(best["per_class_rec"][k])
        ):
            continue

        per_class.append(
            {
                "class": cat_id_to_name[int(cid)],
                "map@50:95": ap_50_95,
                "map@50": ap_50,
                "precision": best["per_class_prec"][k],
                "recall": best["per_class_rec"][k],
                "f1_score": best["per_class_f1"][k],
            }
        )

    per_class.append(
        {
            "class": "all",
            "map@50:95": map_50_95,
            "map@50": map_50,
            "precision": best["macro_precision"],
            "recall": best["macro_recall"],
            "f1_score": best["macro_f1"],
        }
    )

    return {
        "class_map": per_class,
        "map": map_50,
        "precision": best["macro_precision"],
        "recall": best["macro_recall"],
        "f1_score": best["macro_f1"],
    }


def _compute_mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """Compute pairwise boolean-mask IoU between N predictions and M ground truths.

    Args:
        pred_masks: Boolean mask tensor of shape [N, H, W].
        gt_masks: Boolean mask tensor of shape [M, H, W].

    Returns:
        IoU tensor of shape [N, M].
    """
    n = pred_masks.shape[0]
    m = gt_masks.shape[0]
    if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
        h, w = pred_masks.shape[-2:]
        gt_masks = F.interpolate(gt_masks.float().unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)
    pred_flat = pred_masks.bool().view(n, -1).float()  # [N, HW]
    gt_flat = gt_masks.bool().view(m, -1).float()  # [M, HW]
    inter = torch.mm(pred_flat, gt_flat.t())  # [N, M]
    pred_area = pred_flat.sum(dim=1, keepdim=True)  # [N, 1]
    gt_area = gt_flat.sum(dim=1, keepdim=True)  # [M, 1]
    union = pred_area + gt_area.t() - inter  # [N, M]
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))


def _match_single_class(
    pred_scores: torch.Tensor,
    pred_items: torch.Tensor,
    gt_items: torch.Tensor,
    gt_crowd: torch.Tensor,
    iou_threshold: float,
    iou_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Greedy highest-score-first matching for one class in one image.

    Implements the COCO matching algorithm: each GT is matched at most once;
    detections are processed in descending score order; detections matched to
    crowd GTs are marked as ignored rather than false positives.

    Args:
        pred_scores: Float tensor of shape [N] with detection confidences.
        pred_items: Predictions — boxes [N, 4] in xyxy coords or masks [N, H, W].
        gt_items: Ground truths — boxes [M, 4] in xyxy coords or masks [M, H, W].
        gt_crowd: Bool tensor of shape [M], True for crowd instances.
        iou_threshold: Minimum IoU to count as a positive match.
        iou_type: ``"bbox"`` for box IoU or ``"segm"`` for mask IoU.

    Returns:
        Tuple ``(scores_np, matches_np, ignore_np, total_gt)`` where:
            - scores_np: float32 array [N] ordered by descending score.
            - matches_np: int array [N], 1 = TP, 0 = FP.
            - ignore_np: bool array [N], True if matched to a crowd GT.
            - total_gt: number of non-crowd GT instances.
    """
    n = pred_scores.shape[0]
    m = gt_items.shape[0]

    sort_idx = torch.argsort(pred_scores, descending=True)
    pred_scores_sorted = pred_scores[sort_idx]
    pred_sorted = pred_items[sort_idx]

    if iou_type == "bbox":
        iou_matrix = box_iou(pred_sorted, gt_items)  # [N, M]
    else:
        iou_matrix = _compute_mask_iou(pred_sorted, gt_items)  # [N, M]

    device = pred_scores.device
    gt_matched = torch.zeros(m, dtype=torch.bool, device=device)
    pred_match = torch.zeros(n, dtype=torch.long, device=device)
    pred_ignore = torch.zeros(n, dtype=torch.bool, device=device)

    for i in range(n):
        ious = iou_matrix[i]  # [M]

        # Try to match to a non-crowd GT (each non-crowd GT matched at most once).
        nc_ious = ious.clone()
        nc_ious[gt_crowd] = -1.0
        nc_ious[gt_matched & ~gt_crowd] = -1.0  # already claimed

        best_nc_iou, best_nc_idx = nc_ious.max(dim=0)
        if best_nc_iou >= iou_threshold:
            pred_match[i] = 1
            gt_matched[best_nc_idx] = True
        else:
            # A detection matched to a crowd GT is ignored (not a false positive).
            if gt_crowd.any():
                crowd_ious = ious.clone()
                crowd_ious[~gt_crowd] = -1.0
                if crowd_ious.max() >= iou_threshold:
                    pred_ignore[i] = True
            # else: false positive — pred_match stays 0

    total_gt = int((~gt_crowd).sum().item())
    return (
        pred_scores_sorted.float().cpu().numpy().astype(np.float32),
        pred_match.cpu().numpy(),
        pred_ignore.cpu().numpy().astype(bool),
        total_gt,
    )


def build_matching_data(
    preds_list: list[dict[str, torch.Tensor]],
    targets_list: list[dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    iou_type: str = "bbox",
) -> dict[int, dict[str, Any]]:
    """Build compact per-class matching data from a batch of predictions and targets.

    Implements greedy highest-score-first matching compatible with the COCO
    algorithm. The returned dict can be passed directly to
    ``merge_matching_data()`` and ultimately consumed by
    ``sweep_confidence_thresholds()`` after conversion to list form.

    Args:
        preds_list: Per-image predictions. Each dict must contain:

            - ``boxes``: float Tensor [N, 4] in absolute xyxy coordinates.
            - ``scores``: float Tensor [N].
            - ``labels``: int64 Tensor [N].
            - ``masks`` *(optional)*: bool Tensor [N, H, W] for segmentation.

        targets_list: Per-image ground truths. Each dict must contain:

            - ``boxes``: float Tensor [M, 4] in absolute xyxy coordinates.
            - ``labels``: int64 Tensor [M].
            - ``masks`` *(optional)*: bool Tensor [M, H, W] for segmentation.
            - ``iscrowd`` *(optional)*: int64 Tensor [M], 1 for crowd instances.

        iou_threshold: IoU threshold for positive matching. Defaults to 0.5.
        iou_type: ``"bbox"`` for bounding-box IoU; ``"segm"`` for boolean-mask
            IoU. Defaults to ``"bbox"``.

    Returns:
        Dict mapping ``class_id`` (int) to a compact matching dict with keys:

            - ``"scores"``: float32 ndarray of detection scores.
            - ``"matches"``: int ndarray (1 = TP, 0 = FP).
            - ``"ignore"``: bool ndarray (True if matched to a crowd GT).
            - ``"total_gt"``: int, count of non-crowd GT instances.
    """
    acc: dict[int, dict[str, list | int]] = {}

    for preds, targets in zip(preds_list, targets_list):
        pred_boxes = preds["boxes"]  # [N, 4]
        pred_scores = preds["scores"]  # [N]
        pred_labels = preds["labels"]  # [N]
        pred_masks = preds.get("masks")  # [N, H, W] | None

        gt_boxes = targets["boxes"]  # [M, 4]
        gt_labels = targets["labels"]  # [M]
        gt_masks = targets.get("masks")  # [M, H, W] | None
        raw_crowd = targets.get(
            "iscrowd",
            torch.zeros(len(gt_labels), dtype=torch.long, device=gt_labels.device),
        )
        gt_crowd = raw_crowd.bool()

        all_class_ids: set[int] = set(gt_labels.tolist()) | set(pred_labels.tolist())

        for class_id in all_class_ids:
            pred_mask_c = pred_labels == class_id
            gt_mask_c = gt_labels == class_id

            p_scores = pred_scores[pred_mask_c]
            gt_crowd_c = gt_crowd[gt_mask_c]
            n_pred = int(pred_mask_c.sum().item())
            n_gt = int(gt_mask_c.sum().item())

            entry = acc.setdefault(
                class_id,
                {"scores": [], "matches": [], "ignore": [], "total_gt": 0},
            )

            if n_pred == 0:
                entry["total_gt"] += int((~gt_crowd_c).sum().item())
                continue

            if n_gt == 0:
                # TODO: support bfloat16 natively once numpy adds bf16 dtype
                sc = p_scores.float().cpu().numpy()
                order = np.argsort(-sc)
                entry["scores"].extend(sc[order].tolist())
                entry["matches"].extend([0] * n_pred)
                entry["ignore"].extend([False] * n_pred)
                continue

            if iou_type == "bbox":
                p_items: torch.Tensor = pred_boxes[pred_mask_c]  # [n_pred, 4]
                gt_items: torch.Tensor = gt_boxes[gt_mask_c]  # [n_gt, 4]
            else:
                if pred_masks is None or gt_masks is None:
                    raise ValueError("iou_type='segm' requires 'masks' in both preds and targets")
                p_items = pred_masks[pred_mask_c]  # [n_pred, H, W]
                gt_items = gt_masks[gt_mask_c]  # [n_gt, H, W]

            scores_np, matches_np, ignore_np, total_gt = _match_single_class(
                p_scores, p_items, gt_items, gt_crowd_c, iou_threshold, iou_type
            )

            entry["scores"].extend(scores_np.tolist())
            entry["matches"].extend(matches_np.tolist())
            entry["ignore"].extend(ignore_np.tolist())
            entry["total_gt"] += total_gt

    return {
        class_id: {
            "scores": np.array(data["scores"], dtype=np.float32),
            "matches": np.array(data["matches"], dtype=np.int64),
            "ignore": np.array(data["ignore"], dtype=bool),
            "total_gt": data["total_gt"],
        }
        for class_id, data in acc.items()
    }


def init_matching_accumulator() -> dict[int, dict[str, Any]]:
    """Return an empty matching accumulator compatible with ``merge_matching_data()``.

    Returns:
        Empty dict to be passed as the first argument to ``merge_matching_data()``.
    """
    return {}


def merge_matching_data(
    accumulator: dict[int, dict[str, Any]],
    new_data: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Merge *new_data* into *accumulator* in place.

    Both arguments share the dict schema produced by ``build_matching_data()``:
    each class-keyed sub-dict contains ``"scores"`` (float32 ndarray),
    ``"matches"`` (int64 ndarray), ``"ignore"`` (bool ndarray), and
    ``"total_gt"`` (int).

    Args:
        accumulator: Running accumulator, modified in place.
        new_data: Batch-level matching data to merge in.

    Returns:
        The modified *accumulator* (same object, for method chaining).
    """
    for class_id, data in new_data.items():
        if class_id not in accumulator:
            accumulator[class_id] = {
                "scores": data["scores"].copy(),
                "matches": data["matches"].copy(),
                "ignore": data["ignore"].copy(),
                "total_gt": data["total_gt"],
            }
        else:
            entry = accumulator[class_id]
            entry["scores"] = np.concatenate([entry["scores"], data["scores"]])
            entry["matches"] = np.concatenate([entry["matches"], data["matches"]])
            entry["ignore"] = np.concatenate([entry["ignore"], data["ignore"]])
            entry["total_gt"] += data["total_gt"]
    return accumulator


def distributed_merge_matching_data(
    local_data: dict[int, dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Gather per-rank matching data from all DDP ranks and merge into one dict.

    Uses ``utils.all_gather`` (pickle-based) so the data need not be a tensor.
    In single-process (non-distributed) mode, returns a merged copy of *local_data*
    unchanged.

    Args:
        local_data: Per-rank accumulator produced by ``merge_matching_data()``.

    Returns:
        Merged accumulator containing contributions from all ranks.
    """
    gathered: List[dict[int, dict[str, Any]]] = utils.all_gather(local_data)
    merged: dict[int, dict[str, Any]] = {}
    for rank_data in gathered:
        merge_matching_data(merged, rank_data)
    return merged


def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None, header="Eval"):
    model.to(device)
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    iou_types = ("bbox",) if not args.segmentation_head else ("bbox", "segm")
    coco_evaluator = CocoEvaluator(base_ds, iou_types, args.eval_max_dets)

    print_freq = args.print_freq if args is not None else 10
    use_progress_bar = bool(getattr(args, "progress_bar", False))
    if use_progress_bar:
        progress_iter = tqdm(
            data_loader,
            total=len(data_loader),
            desc=header,
            colour="green",
            disable=not utils.is_main_process(),
        )
    else:
        progress_iter = metric_logger.log_every(data_loader, print_freq, header)

    for samples, targets in progress_iter:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        # Add autocast for evaluation
        with autocast(**get_autocast_args(args, device)):
            outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][sub_key].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_all = postprocess(outputs, orig_target_sizes)
        res = {target["image_id"].item(): output for target, output in zip(targets, results_all)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if use_progress_bar:
            log_dict = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            postfix = {
                "class_loss": f"{log_dict['class_error']:.2f}",
                "box_loss": f"{log_dict['loss_bbox']:.2f}",
                "loss": f"{log_dict['loss']:.2f}",
            }
            if _is_cuda(device):
                postfix["max_mem"] = f"{torch.cuda.max_memory_allocated(device) / BYTES_TO_MB:.0f} MB"
            progress_iter.set_postfix(postfix)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Evaluation results: {metric_logger}")
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
        stats["results_json"] = results_json
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

        if "segm" in iou_types:
            results_json_masks = coco_extended_metrics(coco_evaluator.coco_eval["segm"])
            stats["results_json_masks"] = results_json_masks
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator
