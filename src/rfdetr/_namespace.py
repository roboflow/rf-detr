# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Package-private helper: build a self-contained namespace from Pydantic configs.

Replaces the previous shim in ``_args.py`` that called the deprecated
``populate_args()`` function from ``main.py``.  This module has zero dependency
on ``main.py`` and can survive its deletion.
"""

import types
from typing import Any

from rfdetr.config import ModelConfig, TrainConfig


def build_namespace(model_config: ModelConfig, train_config: TrainConfig) -> Any:
    """Build a ``types.SimpleNamespace`` from Pydantic model and train configs.

    Produces the same attribute set as the legacy ``populate_args()`` so that
    ``build_model()``, ``build_criterion_and_postprocessors()``, and
    ``build_dataset()`` continue to work without modification.

    Fields not present in either config retain their ``populate_args()``
    defaults, ensuring downstream consumers see a fully-populated namespace.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.

    Returns:
        ``types.SimpleNamespace`` compatible with ``build_model``,
        ``build_criterion_and_postprocessors``, and ``build_dataset``.
    """
    mc = model_config
    tc = train_config
    train_fields_set = getattr(tc, "model_fields_set", set())
    model_fields_set = getattr(mc, "model_fields_set", set())
    # Transitional compatibility: during deprecation, preserve explicit
    # ModelConfig.cls_loss_coef values when TrainConfig does not set one.
    cls_loss_coef = (
        tc.cls_loss_coef
        if "cls_loss_coef" in train_fields_set or "cls_loss_coef" not in model_fields_set
        else mc.cls_loss_coef
    )

    return types.SimpleNamespace(
        # --- ModelConfig fields ---
        encoder=mc.encoder,
        out_feature_indexes=mc.out_feature_indexes,
        dec_layers=mc.dec_layers,
        freeze_encoder=mc.freeze_encoder,
        backbone_lora=mc.backbone_lora,
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
        cls_loss_coef=cls_loss_coef,
        segmentation_head=mc.segmentation_head,
        mask_downsample_ratio=mc.mask_downsample_ratio,
        num_queries=mc.num_queries,
        num_select=mc.num_select,
        # --- TrainConfig fields ---
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
        # Evaluation extras forwarded via extra_kwargs in the legacy shim.
        eval_max_dets=tc.eval_max_dets,
        eval_interval=tc.eval_interval,
        log_per_class_metrics=tc.log_per_class_metrics,
        compute_val_loss=tc.compute_val_loss,
        compute_test_loss=tc.compute_test_loss,
        ema_update_interval=tc.ema_update_interval,
        train_log_sync_dist=tc.train_log_sync_dist,
        train_log_on_step=tc.train_log_on_step,
        prefetch_factor=tc.prefetch_factor,
        # --- Hardcoded defaults (not in configs; kept for downstream consumers) ---
        print_freq=10,
        clip_max_norm=tc.clip_max_norm,
        do_benchmark=False,
        dropout=0,
        drop_mode="standard",
        drop_schedule="constant",
        cutoff_epoch=0,
        pretrained_encoder=None,
        pretrain_exclude_keys=None,
        pretrain_keys_modify_to_load=None,
        pretrained_distiller=None,
        vit_encoder_num_layers=12,
        window_block_indexes=None,
        position_embedding="sine",
        rms_norm=False,
        force_no_pretrain=False,
        dim_feedforward=2048,
        decoder_norm="LN",
        freeze_batch_norm=False,
        set_cost_class=2,
        set_cost_bbox=5,
        set_cost_giou=2,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        focal_alpha=0.25,
        aux_loss=True,
        sum_group_losses=False,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        coco_path=None,
        aug_config=tc.aug_config,
        dont_save_weights=False,
        seed=tc.seed if tc.seed is not None else 42,
        start_epoch=0,
        eval=False,
        use_ema=tc.use_ema,
        world_size=1,
        dist_url="env://",
        sync_bn=tc.sync_bn,
        fp16_eval=tc.fp16_eval,
        encoder_only=False,
        backbone_only=False,
        use_cls_token=False,
        lr_scheduler="step",
        lr_min_factor=0.0,
        early_stopping=tc.early_stopping,
        early_stopping_patience=tc.early_stopping_patience,
        early_stopping_min_delta=tc.early_stopping_min_delta,
        early_stopping_use_ema=tc.early_stopping_use_ema,
        subcommand=None,
    )
