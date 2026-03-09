# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Package-private helper: map Pydantic configs to the legacy argparse.Namespace.

TODO(Chapter 6): delete this entire module when populate_args() and main.py
    are removed (Phase 9).  Callers in module.py / datamodule.py should then
    read from model_config / train_config directly.
"""

import warnings
from typing import Any

from rfdetr.config import ModelConfig, TrainConfig

# TODO(Chapter 6): remove this import when main.py is deleted.
from rfdetr.main import populate_args


def _build_args_from_configs(model_config: ModelConfig, train_config: TrainConfig) -> Any:
    """Map Pydantic configs to the legacy argparse.Namespace.

    Shared by ``RFDETRModule`` and ``RFDETRDataModule`` so the mapping is
    defined in exactly one place.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.

    Returns:
        Namespace compatible with ``build_model``,
        ``build_criterion_and_postprocessors``, and ``build_dataset``.
    """
    mc = model_config
    tc = train_config
    # This shim intentionally calls the deprecated populate_args() for now.
    # Suppress only its migration warning so users do not see internal noise.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=".*populate_args.*",
        )
        return populate_args(
            # --- ModelConfig ---
            encoder=mc.encoder,
            out_feature_indexes=mc.out_feature_indexes,
            dec_layers=mc.dec_layers,
            two_stage=mc.two_stage,
            projector_scale=mc.projector_scale,
            hidden_dim=mc.hidden_dim,
            patch_size=mc.patch_size,
            num_windows=mc.num_windows,
            sa_nheads=mc.sa_nheads,
            ca_nheads=mc.ca_nheads,
            dec_n_points=mc.dec_n_points,
            bbox_reparam=mc.bbox_reparam,
            lite_refpoint_refine=mc.lite_refpoint_refine,
            layer_norm=mc.layer_norm,
            amp=mc.amp,
            num_classes=mc.num_classes,
            pretrain_weights=mc.pretrain_weights,
            device=mc.device,
            resolution=mc.resolution,
            group_detr=mc.group_detr,
            gradient_checkpointing=mc.gradient_checkpointing,
            positional_encoding_size=mc.positional_encoding_size,
            ia_bce_loss=mc.ia_bce_loss,
            cls_loss_coef=mc.cls_loss_coef,
            segmentation_head=mc.segmentation_head,
            mask_downsample_ratio=mc.mask_downsample_ratio,
            # num_queries / num_select live on subclass configs.
            num_queries=getattr(mc, "num_queries", 300),
            num_select=getattr(mc, "num_select", tc.num_select),
            # --- TrainConfig ---
            lr=tc.lr,
            lr_encoder=tc.lr_encoder,
            batch_size=tc.batch_size,
            grad_accum_steps=tc.grad_accum_steps,
            epochs=tc.epochs,
            resume=tc.resume or "",
            ema_decay=tc.ema_decay,
            ema_tau=tc.ema_tau,
            lr_drop=tc.lr_drop,
            checkpoint_interval=tc.checkpoint_interval,
            warmup_epochs=tc.warmup_epochs,
            lr_vit_layer_decay=tc.lr_vit_layer_decay,
            lr_component_decay=tc.lr_component_decay,
            drop_path=tc.drop_path,
            weight_decay=tc.weight_decay,
            multi_scale=tc.multi_scale,
            expanded_scales=tc.expanded_scales,
            do_random_resize_via_padding=tc.do_random_resize_via_padding,
            square_resize_div_64=tc.square_resize_div_64,
            num_workers=tc.num_workers,
            dataset_file=tc.dataset_file,
            dataset_dir=tc.dataset_dir,
            output_dir=tc.output_dir,
            # Segmentation extras (present on SegmentationTrainConfig only).
            mask_ce_loss_coef=getattr(tc, "mask_ce_loss_coef", 5.0),
            mask_dice_loss_coef=getattr(tc, "mask_dice_loss_coef", 5.0),
            mask_point_sample_ratio=getattr(tc, "mask_point_sample_ratio", 16),
            # Evaluation extras — not in populate_args' named parameters but
            # accepted via **extra_kwargs and forwarded into argparse.Namespace.
            eval_max_dets=tc.eval_max_dets,
            eval_interval=tc.eval_interval,
            log_per_class_metrics=tc.log_per_class_metrics,
            compute_val_loss=tc.compute_val_loss,
            compute_test_loss=tc.compute_test_loss,
            ema_update_interval=tc.ema_update_interval,
            train_log_sync_dist=tc.train_log_sync_dist,
            train_log_on_step=tc.train_log_on_step,
            prefetch_factor=tc.prefetch_factor,
        )
