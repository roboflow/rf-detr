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
import warnings

from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.models._defaults import MODEL_DEFAULTS, ModelDefaults


def _namespace_from_configs(
    model_config: ModelConfig,
    train_config: TrainConfig,
    defaults: ModelDefaults = MODEL_DEFAULTS,
) -> types.SimpleNamespace:
    """Build a ``types.SimpleNamespace`` from configs and architectural defaults.

    This is the internal implementation behind :func:`build_namespace`.
    Extracting it allows config-native builder functions to construct a
    namespace without going through the public ``build_namespace()`` API
    while still accepting overridable defaults.

    This function is used by multiple modules as the transitional namespace
    bridge: :func:`rfdetr.models.build_model_from_config`,
    :func:`rfdetr.models.build_criterion_from_config`, and
    :func:`rfdetr.detr._build_model_context` all call it directly to avoid
    the public ``build_namespace()`` shim.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.
        defaults: Hardcoded architectural constants.  Defaults to
            :data:`MODEL_DEFAULTS`.

    Returns:
        ``types.SimpleNamespace`` compatible with ``build_model``,
        ``build_criterion_and_postprocessors``, and ``build_dataset``.
    """
    mc = model_config
    tc = train_config
    d = defaults
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
        # --- Hardcoded defaults (from ModelDefaults) ---
        print_freq=d.print_freq,
        clip_max_norm=tc.clip_max_norm,
        do_benchmark=d.do_benchmark,
        dropout=d.dropout,
        drop_mode=d.drop_mode,
        drop_schedule=d.drop_schedule,
        cutoff_epoch=d.cutoff_epoch,
        pretrained_encoder=d.pretrained_encoder,
        pretrain_exclude_keys=d.pretrain_exclude_keys,
        pretrain_keys_modify_to_load=d.pretrain_keys_modify_to_load,
        pretrained_distiller=d.pretrained_distiller,
        vit_encoder_num_layers=d.vit_encoder_num_layers,
        window_block_indexes=d.window_block_indexes,
        position_embedding=d.position_embedding,
        rms_norm=d.rms_norm,
        force_no_pretrain=d.force_no_pretrain,
        dim_feedforward=d.dim_feedforward,
        decoder_norm=d.decoder_norm,
        freeze_batch_norm=d.freeze_batch_norm,
        set_cost_class=d.set_cost_class,
        set_cost_bbox=d.set_cost_bbox,
        set_cost_giou=d.set_cost_giou,
        bbox_loss_coef=d.bbox_loss_coef,
        giou_loss_coef=d.giou_loss_coef,
        focal_alpha=d.focal_alpha,
        aux_loss=d.aux_loss,
        sum_group_losses=d.sum_group_losses,
        use_varifocal_loss=d.use_varifocal_loss,
        use_position_supervised_loss=d.use_position_supervised_loss,
        coco_path=d.coco_path,
        aug_config=tc.aug_config,
        dont_save_weights=d.dont_save_weights,
        seed=tc.seed if tc.seed is not None else 42,
        start_epoch=d.start_epoch,
        eval=d.eval,
        use_ema=tc.use_ema,
        world_size=d.world_size,
        dist_url=d.dist_url,
        sync_bn=tc.sync_bn,
        fp16_eval=tc.fp16_eval,
        encoder_only=d.encoder_only,
        backbone_only=d.backbone_only,
        use_cls_token=d.use_cls_token,
        lr_scheduler=d.lr_scheduler,
        lr_min_factor=d.lr_min_factor,
        early_stopping=tc.early_stopping,
        early_stopping_patience=tc.early_stopping_patience,
        early_stopping_min_delta=tc.early_stopping_min_delta,
        early_stopping_use_ema=tc.early_stopping_use_ema,
        subcommand=d.subcommand,
    )


def build_namespace(model_config: ModelConfig, train_config: TrainConfig) -> types.SimpleNamespace:
    """Build a ``types.SimpleNamespace`` from Pydantic model and train configs.

    .. deprecated::
        ``build_namespace`` is a backward-compatibility shim with no remaining
        internal callers.  Use the config-native builders instead:

        - :func:`rfdetr.models.build_model_from_config` — replaces
          ``build_model(build_namespace(mc, tc))``
        - :func:`rfdetr.models.build_criterion_from_config` — replaces
          ``build_criterion_and_postprocessors(build_namespace(mc, tc))``
        - :func:`rfdetr._namespace._namespace_from_configs` — for the rare
          case where a raw namespace is still required (e.g. ``build_dataset``)

        ``build_namespace`` will be removed in v1.9.

    Args:
        model_config: Architecture configuration.
        train_config: Training hyperparameter configuration.

    Returns:
        ``types.SimpleNamespace`` compatible with ``build_model``,
        ``build_criterion_and_postprocessors``, and ``build_dataset``.
    """
    warnings.warn(
        "build_namespace() is deprecated and will be removed in v1.9. "
        "Use build_model_from_config() or build_criterion_from_config() instead; "
        "for raw namespace access use rfdetr._namespace._namespace_from_configs().",
        DeprecationWarning,
        stacklevel=2,
    )
    return _namespace_from_configs(model_config, train_config)
