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


def get_autocast_args(args):
    if DEPRECATED_AMP:
        return {'enabled': args.amp, 'dtype': torch.bfloat16}
    else:
        return {'device_type': 'cuda', 'enabled': args.amp, 'dtype': torch.bfloat16}


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
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = args.print_freq if args is not None else 10
    start_steps = epoch * num_training_steps_per_epoch

    # Add gradient scaler for AMP
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler('cuda', enabled=args.amp)

    optimizer.zero_grad()

    # Check if batch size is divisible by gradient accumulation steps
    if batch_size % args.grad_accum_steps != 0:
        logger.error(f"Batch size ({batch_size}) must be divisible by gradient accumulation steps ({args.grad_accum_steps})")
        raise ValueError(f"Batch size ({batch_size}) must be divisible by gradient accumulation steps ({args.grad_accum_steps})")

    logger.info(f"Training config: grad_accum_steps={args.grad_accum_steps}, "
                f"total_batch_size={batch_size * get_world_size()}, "
                f"dataloader_length={len(data_loader)}")

    sub_batch_size = batch_size // args.grad_accum_steps

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
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
                model.module.update_drop_path(
                    schedules["dp"][it], vit_encoder_num_layers
                )
            else:
                model.update_drop_path(schedules["dp"][it], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][it])
            else:
                model.update_dropout(schedules["do"][it])

        if args.multi_scale and not args.do_random_resize_via_padding:
            scales = compute_multi_scale_scales(args.resolution, args.expanded_scales, args.patch_size, args.num_windows)
            random.seed(it)
            scale = random.choice(scales)
            with torch.no_grad():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode='bilinear', align_corners=False)
                samples.mask = F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode='nearest').squeeze(1).bool()

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
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k:  v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
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
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Epoch {epoch} stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def coco_extended_metrics(coco_eval):
    """
    Compute precision/recall/F1 using COCO precision arrays.

    Uses eval["precision"] and eval["scores"] arrays from the COCO evaluator,
    which are always populated regardless of faster-coco-eval's extra_calc setting.
    Finds the recall threshold that maximises macro-F1 and reports metrics there.
    """
    iou_thrs = coco_eval.params.iouThrs
    rec_thrs = coco_eval.params.recThrs

    # Safely locate the IoU=0.50 slice index
    iou50_indices = np.where(np.isclose(iou_thrs, 0.50))[0]
    iou50_idx = int(iou50_indices[0]) if len(iou50_indices) > 0 else None

    area_idx = 0
    maxdet_idx = -1  # last maxDet entry

    P = coco_eval.eval["precision"]  # [T, R, K, A, M]

    # Precision at IoU=0.50 across all recall thresholds and classes: [R, K]
    if iou50_idx is not None:
        prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx]
    else:
        prec_raw = np.full((P.shape[1], P.shape[2]), -1.0)

    prec = prec_raw.copy().astype(float)
    prec[prec < 0] = np.nan  # -1 sentinels → NaN

    # F1 at each recall threshold for each class; guard 0/0 with np.where
    denom = prec + rec_thrs[:, None]
    f1_cls = np.where(denom > 0, 2 * prec * rec_thrs[:, None] / denom, 0.0)
    f1_macro = np.nanmean(f1_cls, axis=1)  # [R] macro-F1 per recall threshold

    # Best recall threshold (robust to all-NaN / all-zero degenerate cases)
    if np.all(np.isnan(f1_macro)) or float(np.nanmax(f1_macro)) == 0.0:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        best_j = 0
    else:
        best_j = int(np.nanargmax(f1_macro))
        prec_at_best = prec[best_j]
        macro_precision = float(np.nanmean(prec_at_best)) if not np.all(np.isnan(prec_at_best)) else 0.0
        macro_recall = float(rec_thrs[best_j])
        macro_f1 = float(f1_macro[best_j])

    map_50_95 = float(coco_eval.stats[0])
    map_50 = float(coco_eval.stats[1])

    per_class = []
    cat_ids = coco_eval.params.catIds
    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}

    for k, cid in enumerate(cat_ids):
        p_slice = P[:, :, k, area_idx, maxdet_idx]  # [T, R]
        p_masked = np.where(p_slice > -1, p_slice, np.nan)

        # Two-pass nanmean to weight each IoU threshold equally (standard COCO AP)
        ap_50_95 = float(np.nanmean(np.nanmean(p_masked, axis=1)))

        if iou50_idx is not None and (p_slice[iou50_idx] > -1).any():
            ap_50 = float(np.nanmean(p_masked[iou50_idx]))
        else:
            ap_50 = float("nan")

        pc = float(prec[best_j, k]) if not np.isnan(prec[best_j, k]) else float("nan")
        rc = macro_recall
        f1c = float(f1_cls[best_j, k])

        if np.isnan(ap_50_95) or np.isnan(ap_50) or np.isnan(pc):
            continue

        per_class.append({
            "class"     : cat_id_to_name[int(cid)],
            "map@50:95" : ap_50_95,
            "map@50"    : ap_50,
            "precision" : pc,
            "recall"    : rc,
            "f1_score"  : f1c,
        })

    per_class.append({
        "class"     : "all",
        "map@50:95" : map_50_95,
        "map@50"    : map_50,
        "precision" : macro_precision,
        "recall"    : macro_recall,
        "f1_score"  : macro_f1,
    })

    return {
        "class_map": per_class,
        "map"      : map_50,
        "precision": macro_precision,
        "recall"   : macro_recall,
        "f1_score" : macro_f1,
    }

def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"

    iou_types = ("bbox",) if not args.segmentation_head else ("bbox", "segm")
    coco_evaluator = CocoEvaluator(base_ds, iou_types, args.eval_max_dets)

    print_freq = args.print_freq if args is not None else 10
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
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
                            outputs[key][idx][sub_key] = outputs[key][idx][
                                sub_key
                            ].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results_all = postprocess(outputs, orig_target_sizes)
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, results_all)
        }
        if coco_evaluator is not None:
            coco_evaluator.update(res)

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
