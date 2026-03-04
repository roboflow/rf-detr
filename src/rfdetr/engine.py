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
from typing import Callable, DefaultDict, List

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


def _get_cuda_autocast_dtype() -> torch.dtype:
    """Return the autocast dtype that is supported on the current CUDA device."""
    if not torch.cuda.is_available():
        return torch.bfloat16

    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(is_bf16_supported):
        return torch.bfloat16 if is_bf16_supported() else torch.float16

    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def get_autocast_args(args):
    autocast_dtype = _get_cuda_autocast_dtype()
    if DEPRECATED_AMP:
        return {"enabled": args.amp, "dtype": autocast_dtype}
    else:
        return {"device_type": "cuda", "enabled": args.amp, "dtype": autocast_dtype}


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
        scaler = GradScaler("cuda", enabled=args.amp)

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

            with autocast(**get_autocast_args(args)):
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


def coco_extended_metrics(coco_eval):
    """
    Compute precision/recall by sweeping confidence thresholds to maximize macro-F1.
    Uses evalImgs directly to compute metrics from raw matching data.
    """

    iou50_idx = np.argmax(np.isclose(coco_eval.params.iouThrs, 0.50)).item()
    cat_ids = coco_eval.params.catIds
    num_classes = len(cat_ids)
    area_idx = 0
    maxdet_idx = 2

    # Unflatten evalImgs into a nested dict
    evalImgs_unflat = {}
    for e in coco_eval.evalImgs:
        if e is None:
            continue
        cat_id = e["category_id"]
        area_rng = tuple(e["aRng"])
        img_id = e["image_id"]

        if cat_id not in evalImgs_unflat:
            evalImgs_unflat[cat_id] = {}
        if area_rng not in evalImgs_unflat[cat_id]:
            evalImgs_unflat[cat_id][area_rng] = {}
        evalImgs_unflat[cat_id][area_rng][img_id] = e

    area_rng_all = tuple(coco_eval.params.areaRng[area_idx])

    per_class_data = []
    for cid in cat_ids:
        dt_scores = []
        dt_matches = []
        dt_ignore = []
        total_gt = 0

        for img_id in coco_eval.params.imgIds:
            e = evalImgs_unflat.get(cid, {}).get(area_rng_all, {}).get(img_id)
            if e is None:
                continue

            num_dt = len(e["dtIds"])
            # num_gt = len(e['gtIds'])

            gt_ignore = e["gtIgnore"]
            total_gt += sum(1 for ig in gt_ignore if not ig)

            for d in range(num_dt):
                dt_scores.append(e["dtScores"][d])
                dt_matches.append(e["dtMatches"][iou50_idx, d])
                dt_ignore.append(e["dtIgnore"][iou50_idx, d])

        per_class_data.append(
            {
                "scores": np.array(dt_scores),
                "matches": np.array(dt_matches),
                "ignore": np.array(dt_ignore, dtype=bool),
                "total_gt": total_gt,
            }
        )

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

        # We do this as two sequential nanmeans to avoid
        # underweighting columns with more nans, since each
        # column corresponds to a different IoU threshold
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


def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None, header="Eval"):
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
        with autocast(**get_autocast_args(args)):
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
