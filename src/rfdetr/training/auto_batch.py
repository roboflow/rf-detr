from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch

from rfdetr._namespace import build_namespace
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.models.lwdetr import build_criterion_and_postprocessors
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.tensors import NestedTensor

logger = get_logger()


@dataclass(frozen=True)
class AutoBatchResult:
    safe_micro_batch: int
    recommended_grad_accum_steps: int
    effective_batch_size: int
    device_name: str


def _is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _make_synthetic_batch(
    micro_batch_size: int,
    resolution: int,
    device: torch.device,
    num_classes: int,
) -> Tuple[NestedTensor, List[Dict[str, torch.Tensor]]]:
    tensors = torch.randn(micro_batch_size, 3, resolution, resolution, device=device)
    mask = torch.zeros(micro_batch_size, resolution, resolution, dtype=torch.bool, device=device)
    samples = NestedTensor(tensors, mask)

    max_label = max(0, num_classes - 1)
    targets: List[Dict[str, torch.Tensor]] = []
    for idx in range(micro_batch_size):
        targets.append(
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32, device=device),
                "labels": torch.tensor([min(1, max_label)], dtype=torch.int64, device=device),
                "image_id": torch.tensor(idx, dtype=torch.int64, device=device),
                "orig_size": torch.tensor([resolution, resolution], dtype=torch.int64, device=device),
                "size": torch.tensor([resolution, resolution], dtype=torch.int64, device=device),
                "iscrowd": torch.tensor([0], dtype=torch.int64, device=device),
                "area": torch.tensor([0.04], dtype=torch.float32, device=device),
            }
        )
    return samples, targets


def _probe_step(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    micro_batch_size: int,
    resolution: int,
    device: torch.device,
    num_classes: int,
    amp: bool,
) -> bool:
    try:
        model.zero_grad(set_to_none=True)
        criterion.zero_grad(set_to_none=True)
        samples, targets = _make_synthetic_batch(
            micro_batch_size=micro_batch_size,
            resolution=resolution,
            device=device,
            num_classes=num_classes,
        )

        with torch.autocast(device_type="cuda", enabled=amp):
            outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss = sum(loss_dict[name] * weight_dict[name] for name in loss_dict if name in weight_dict)

        if not torch.isfinite(loss):
            raise RuntimeError("auto-batch probe produced a non-finite training loss.")

        loss.backward()
        model.zero_grad(set_to_none=True)
        criterion.zero_grad(set_to_none=True)
        return True
    except RuntimeError as exc:
        if _is_cuda_oom(exc):
            torch.cuda.empty_cache()
            return False
        raise


def probe_max_micro_batch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    resolution: int,
    device: torch.device,
    num_classes: int,
    amp: bool,
    safety_margin: float = 0.9,
    max_micro_batch: int = 128,
) -> int:
    if device.type != "cuda":
        raise RuntimeError("auto-batch probing currently supports CUDA only.")
    if max_micro_batch < 1:
        raise ValueError("max_micro_batch must be >= 1.")
    if not (0 < safety_margin <= 1.0):
        raise ValueError("safety_margin must be in (0, 1].")

    model_training = model.training
    criterion_training = criterion.training
    model.train()
    criterion.train()

    try:
        lower_ok = 0
        candidate = 1
        upper_fail = None

        while candidate <= max_micro_batch:
            if _probe_step(model, criterion, candidate, resolution, device, num_classes, amp):
                lower_ok = candidate
                candidate *= 2
            else:
                upper_fail = candidate
                break

        if lower_ok < 1:
            raise RuntimeError(
                "auto-batch probe failed at micro_batch_size=1. "
                "Try lowering resolution or enabling gradient_checkpointing."
            )

        if upper_fail is None:
            upper_fail = max_micro_batch + 1

        lo = lower_ok + 1
        hi = min(upper_fail - 1, max_micro_batch)
        while lo <= hi:
            mid = (lo + hi) // 2
            if _probe_step(model, criterion, mid, resolution, device, num_classes, amp):
                lower_ok = mid
                lo = mid + 1
            else:
                hi = mid - 1

        safe_micro_batch = max(1, math.floor(lower_ok * safety_margin))
        return min(safe_micro_batch, lower_ok)
    finally:
        model.train(model_training)
        criterion.train(criterion_training)
        model.zero_grad(set_to_none=True)
        criterion.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()


def recommend_grad_accum_steps(safe_micro_batch: int, target_effective_batch: int) -> int:
    if safe_micro_batch < 1:
        raise ValueError("safe_micro_batch must be >= 1.")
    if target_effective_batch < 1:
        raise ValueError("target_effective_batch must be >= 1.")
    return max(1, math.ceil(target_effective_batch / safe_micro_batch))


def resolve_auto_batch_config(
    model_context: Any,
    model_config: ModelConfig,
    train_config: TrainConfig,
    safety_margin: float = 0.9,
    max_micro_batch: int = 128,
) -> AutoBatchResult:
    device = model_context.device
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("batch_size='auto' requires a CUDA device for probing in v1.")

    args = build_namespace(model_config, train_config)
    criterion, _ = build_criterion_and_postprocessors(args)
    criterion = criterion.to(device)

    safe_micro_batch = probe_max_micro_batch(
        model=model_context.model,
        criterion=criterion,
        resolution=model_config.resolution,
        device=device,
        num_classes=model_config.num_classes,
        amp=bool(model_config.amp),
        safety_margin=safety_margin,
        max_micro_batch=max_micro_batch,
    )
    grad_accum_steps = recommend_grad_accum_steps(safe_micro_batch, train_config.auto_batch_target_effective)
    effective_batch_size = safe_micro_batch * grad_accum_steps
    device_name = torch.cuda.get_device_name(device)

    logger.info("[auto-batch] device=%s segmentation=%s resolution=%s", device_name, model_config.segmentation_head, model_config.resolution)
    logger.info(
        "[auto-batch] safe_micro_batch=%s grad_accum_steps=%s effective_batch_size=%s",
        safe_micro_batch,
        grad_accum_steps,
        effective_batch_size,
    )
    logger.info("[auto-batch] This probe estimates train-step-safe micro-batch only.")
    logger.info("[auto-batch] Validation/test (especially segmentation mask eval) may require more memory.")

    return AutoBatchResult(
        safe_micro_batch=safe_micro_batch,
        recommended_grad_accum_steps=grad_accum_steps,
        effective_batch_size=effective_batch_size,
        device_name=device_name,
    )
