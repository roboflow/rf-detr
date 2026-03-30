# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""LightningModule for RF-DETR training and validation."""

from __future__ import annotations

import math
import random
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, seed_everything

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.datasets.coco import compute_multi_scale_scales
from rfdetr.models import build_criterion_from_config, build_model_from_config
from rfdetr.models.weights import apply_lora, load_pretrain_weights
from rfdetr.training.param_groups import get_param_dict
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class RFDETRModelModule(LightningModule):
    """LightningModule wrapping the RF-DETR model and training loop.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.
    """

    def __init__(self, model_config: ModelConfig, train_config: TrainConfig) -> None:
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        # Allow partial state-dict loading when resuming from a .pth checkpoint
        # (which contains only model weights, not criterion/postprocess state).
        self.strict_loading = False

        # Model, criterion, and postprocessor.
        self.model = build_model_from_config(model_config, train_config)
        if model_config.pretrain_weights is not None:
            # Capture the configured class count before loading weights so we can
            # detect any automatic alignment to the checkpoint.
            prev_num_classes = self.model_config.num_classes
            load_pretrain_weights(self.model, self.model_config)
            # If the loaded checkpoint changed the model's effective number of
            # classes (e.g. to match a fine-tuned head), persist that back onto
            # the model_config so downstream components see the aligned value.
            if hasattr(self.model, "num_classes"):
                model_num_classes = getattr(self.model, "num_classes")
                if model_num_classes is not None and model_num_classes != prev_num_classes:
                    self.model_config.num_classes = model_num_classes
        if model_config.backbone_lora:
            apply_lora(self.model)

        # Build criterion and postprocessor after potential num_classes
        # alignment so they match the current model head.
        self.criterion, self.postprocess = build_criterion_from_config(self.model_config, self.train_config)

        # torch.compile is opt-in: set model_config.compile=True to enable.
        # Only enabled on CUDA; MPS and CPU do not benefit from compilation.
        compile_enabled = model_config.compile and torch.cuda.is_available() and not train_config.multi_scale
        if model_config.compile and train_config.multi_scale:
            logger.info("Disabling torch.compile because multi_scale=True introduces dynamic input shapes.")
        if compile_enabled:
            # dynamic=True: one compiled graph handles all multi-scale input sizes instead
            # of recompiling per (H, W) pair. suppress_errors=True: if inductor can't
            # compile a subgraph (e.g. bicubic backward with symbolic shapes), it falls
            # back to eager mode for that subgraph rather than crashing.
            # capture_scalar_outputs=True: include Tensor.item() calls
            # (gen_encoder_output_proposals / ms_deform_attn use spatial-shape .item()
            # as Python slice indices). Safe with dynamic=True because item() results
            # are backed symbols derived from input shapes — not unbacked symbols that
            # would cause PendingUnbackedSymbolNotFound (which only occurs without dynamic).
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.capture_scalar_outputs = True
            self.model = torch.compile(self.model, dynamic=True)

    # ------------------------------------------------------------------
    # PTL lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self) -> None:
        """Seed RNGs at fit start when ``TrainConfig.seed`` is set.

        This avoids hidden global side-effects in ``build_trainer`` while still
        preserving deterministic training behaviour for actual fit runs.
        """
        if self.train_config.seed is not None:
            seed_everything(self.train_config.seed + self.global_rank, workers=True)

    def on_train_batch_start(self, batch: Tuple, batch_idx: int) -> None:
        """Apply optional multi-scale resize to the incoming batch.

        Modifications to ``batch`` (in-place on ``NestedTensor``) are visible
        in ``training_step`` because they share the same object.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Index of the current batch within the epoch.
        """
        tc = self.train_config
        mc = self.model_config

        if tc.multi_scale and not tc.do_random_resize_via_padding:
            samples, _ = batch
            scales = compute_multi_scale_scales(mc.resolution, tc.expanded_scales, mc.patch_size, mc.num_windows)
            step = self.trainer.global_step
            random.seed(step)
            scale = random.choice(scales)
            with torch.no_grad():
                samples.tensors = F.interpolate(samples.tensors, size=scale, mode="bilinear", align_corners=False)
                samples.mask = (
                    F.interpolate(samples.mask.unsqueeze(1).float(), size=scale, mode="nearest").squeeze(1).bool()
                )

    def training_step(self, batch: Tuple, batch_idx: int) -> torch.Tensor:
        """Compute loss for one training step and log metrics.

        PTL handles gradient accumulation (``accumulate_grad_batches``), AMP
        (``precision``), and gradient clipping (``gradient_clip_val``) — no
        manual ``GradScaler`` or loss scaling here.  The loss is divided by
        ``trainer.accumulate_grad_batches`` so that the accumulated gradient
        magnitude matches the legacy engine (which scales each sub-batch by
        ``1/grad_accum_steps`` before calling ``backward()``).

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the epoch.

        Returns:
            Scalar loss tensor.
        """
        samples, targets = batch
        batch_size = len(targets)
        outputs = self.model(samples, targets)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
        # Normalise by grad-accum steps so the accumulated gradient matches the
        # legacy engine, which scales each sub-batch by 1/grad_accum_steps before
        # backward().  PTL accumulates full-scale gradients by default; dividing
        # here keeps the effective LR identical to the non-PTL training path.
        # We return the scaled loss to PTL but log the unscaled value so that
        # train/loss and val/loss are on the same scale.
        loss_scaled = loss / self.trainer.accumulate_grad_batches
        train_log_sync_dist = bool(self.train_config.train_log_sync_dist)
        train_log_on_step = bool(self.train_config.train_log_on_step)
        self.log_dict(
            {f"train/{k}": v for k, v in loss_dict.items()},
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=train_log_on_step,
            on_epoch=True,
            sync_dist=train_log_sync_dist,
            batch_size=batch_size,
        )
        optimizer = self.optimizers()
        if isinstance(optimizer, list):
            optimizer = optimizer[0]
        # Optimizer may have multiple param groups with different LRs (e.g., backbone/decoder).
        # Preserve the first group's LR for backward compatibility, but also log the
        # min/max across all groups so the progress bar reflects the full schedule.
        group_lrs = [pg["lr"] for pg in optimizer.param_groups if "lr" in pg]
        if group_lrs:
            base_lr = group_lrs[0]
            min_lr = min(group_lrs)
            max_lr = max(group_lrs)
            self.log("train/lr", base_lr, prog_bar=True, on_step=True, on_epoch=False)
            self.log("train/lr_min", min_lr, prog_bar=True, on_step=True, on_epoch=False)
            self.log("train/lr_max", max_lr, prog_bar=True, on_step=True, on_epoch=False)
        return loss_scaled

    def validation_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Any]:
        """Run forward pass and postprocess for one validation step.

        Returns raw results and targets so ``COCOEvalCallback`` can accumulate
        them across the epoch via ``on_validation_batch_end``.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the validation epoch.

        Returns:
            Dict with ``results`` (postprocessed predictions) and ``targets``.
        """
        samples, targets = batch
        outputs = self.model(samples)
        if self.train_config.compute_val_loss:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=len(targets))

        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = self.postprocess(outputs, orig_sizes)
        return {"results": results, "targets": targets}

    def configure_optimizers(self) -> Dict[str, Any]:
        """Build AdamW optimizer with layer-wise LR decay and LambdaLR scheduler.

        Uses ``trainer.estimated_stepping_batches`` for total step count so
        cosine annealing covers the full training run regardless of dataset
        size or accumulation settings.

        Returns:
            PTL optimizer config dict with optimizer and step-interval scheduler.
        """
        tc = self.train_config
        ns = _namespace_from_configs(self.model_config, tc)

        # Unwrap torch.compile's OptimizedModule so get_param_dict sees the
        # original module's named_parameters() — compiled wrapper can cause
        # name-prefix mismatches that put the same tensor in multiple groups.
        model_for_params = getattr(self.model, "_orig_mod", self.model)
        param_dicts = get_param_dict(ns, model_for_params)
        param_dicts = [p for p in param_dicts if p["params"].requires_grad]
        use_fused = self.model_config.fused_optimizer and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=tc.lr,
            weight_decay=tc.weight_decay,
            fused=use_fused,
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = max(1, total_steps // tc.epochs)
        warmup_steps = int(steps_per_epoch * tc.warmup_epochs)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if tc.lr_scheduler == "cosine":
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return tc.lr_min_factor + (1 - tc.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
            # Step decay: drop by 10× after lr_drop epochs.
            if current_step < tc.lr_drop * steps_per_epoch:
                return 1.0
            return 0.1

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def clip_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        """Override PTL gradient clipping to support fused AdamW.

        PTL's AMP precision plugin refuses to clip gradients when the optimizer
        declares it handles unscaling internally (fused=True).  When fused is
        active we are on BF16 (no GradScaler) so ``clip_grad_norm_`` is
        correct.  For the non-fused path (FP16 + GradScaler or FP32) we
        delegate to ``super()`` to preserve scaler-aware unscaling.

        Args:
            optimizer: The current optimizer.
            gradient_clip_val: Maximum gradient norm.
            gradient_clip_algorithm: Clipping algorithm; forwarded to super()
                for the non-fused path.
        """
        use_fused = self.model_config.fused_optimizer and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if use_fused:
            if gradient_clip_val and gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), gradient_clip_val)
        else:
            super().clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )

    def test_step(self, batch: Tuple, batch_idx: int) -> Dict[str, Any]:
        """Run forward pass and postprocess for one test step.

        Mirrors :meth:`validation_step` so ``COCOEvalCallback`` can accumulate
        results via ``on_test_batch_end`` when ``trainer.test()`` is called (e.g.
        from :class:`~rfdetr.training.callbacks.BestModelCallback` at end of training).

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index within the test epoch.

        Returns:
            Dict with ``results`` (postprocessed predictions) and ``targets``.
        """
        samples, targets = batch
        outputs = self.model(samples)
        if self.train_config.compute_test_loss:
            loss_dict = self.criterion(outputs, targets)
            weight_dict = self.criterion.weight_dict
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
            self.log("test/loss", loss, sync_dist=True, batch_size=len(targets))

        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = self.postprocess(outputs, orig_sizes)
        return {"results": results, "targets": targets}

    def predict_step(self, batch: Tuple, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """Run inference on a preprocessed batch and return postprocessed results.

        Args:
            batch: Tuple of (NestedTensor samples, list of target dicts).
            batch_idx: Batch index.
            dataloader_idx: Index of the predict dataloader.

        Returns:
            Postprocessed detection results from ``PostProcess``.
        """
        samples, targets = batch
        with torch.no_grad():
            outputs = self.model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        return self.postprocess(outputs, orig_sizes)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Auto-detect and normalise legacy ``.pth`` checkpoints at load time.

        PTL calls this hook before applying ``checkpoint["state_dict"]`` to
        the module.  Two legacy formats are handled:

        1. **Raw legacy format** — a ``*.pth`` file loaded directly by
           ``Trainer`` (e.g. via ``ckpt_path=``).  Recognised by the presence
           of ``"model"`` without ``"state_dict"``.  The state dict is
           rewritten in-place with the ``"model."`` prefix so PTL can apply it
           normally.

        2. **Converted format** — a file produced by
           :func:`~rfdetr.training.checkpoint.convert_legacy_checkpoint` that
           already has ``"state_dict"`` but also carries
           ``"legacy_ema_state_dict"``.  The EMA weights are stashed on
           ``self._pending_legacy_ema_state`` for optional restoration by
           :class:`~rfdetr.training.callbacks.ema.RFDETREMACallback`.

        Args:
            checkpoint: Checkpoint dict passed in by PTL (mutated in-place).
        """
        # Raw legacy .pth: no "state_dict" key — build it from "model".
        if "model" in checkpoint and "state_dict" not in checkpoint:
            checkpoint["state_dict"] = {"model." + k: v for k, v in checkpoint["model"].items()}

        # Stash legacy EMA weights for RFDETREMACallback.setup(), which restores
        # them into AveragedModel when resuming from converted legacy checkpoints.
        if "legacy_ema_state_dict" in checkpoint:
            self._pending_legacy_ema_state = checkpoint["legacy_ema_state_dict"]
            warnings.warn(
                "Checkpoint contains legacy EMA weights (`legacy_ema_state_dict`). "
                "Add RFDETREMACallback to your trainer callbacks to restore them; "
                "without it the stashed weights will be ignored.",
                UserWarning,
                stacklevel=2,
            )

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Reinitialize the detection head for a new class count.

        Args:
            num_classes: New number of classes (excluding background).
        """
        self.model.reinitialize_detection_head(num_classes)
