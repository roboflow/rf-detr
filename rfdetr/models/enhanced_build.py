"""
Enhanced model building functions with Color Attention, Enhanced Deformable Attention,
Boundary Refinement, and Color Contrast Loss.
"""

import torch
import torch.nn as nn
from rfdetr.models.backbone import build_backbone
from rfdetr.models.matcher import build_matcher
from rfdetr.models.transformer import build_transformer
from rfdetr.models.enhanced_segmentation_head import EnhancedSegmentationHead
from rfdetr.models.enhanced_criterion import EnhancedSetCriterion
from rfdetr.models.lwdetr import LWDETR, PostProcess


def build_enhanced_model(args):
    """
    Build enhanced RF-DETR model with improved modules:
    - Color Attention Module
    - Boundary Refinement Network
    - Enhanced Deformable Attention (optional)
    """
    num_classes = args.num_classes + 1
    device = torch.device(args.device)

    # Build backbone (same as standard model)
    backbone = build_backbone(
        encoder=args.encoder,
        vit_encoder_num_layers=args.vit_encoder_num_layers,
        pretrained_encoder=args.pretrained_encoder,
        window_block_indexes=args.window_block_indexes,
        drop_path=args.drop_path,
        out_channels=args.hidden_dim,
        out_feature_indexes=args.out_feature_indexes,
        projector_scale=args.projector_scale,
        use_cls_token=args.use_cls_token,
        hidden_dim=args.hidden_dim,
        position_embedding=args.position_embedding,
        freeze_encoder=args.freeze_encoder,
        layer_norm=args.layer_norm,
        target_shape=args.shape if hasattr(args, 'shape') else (args.resolution, args.resolution) if hasattr(args, 'resolution') else (640, 640),
        rms_norm=args.rms_norm,
        backbone_lora=args.backbone_lora,
        force_no_pretrain=args.force_no_pretrain,
        gradient_checkpointing=args.gradient_checkpointing,
        load_dinov2_weights=args.pretrain_weights is None,
        patch_size=args.patch_size,
        num_windows=args.num_windows,
        positional_encoding_size=args.positional_encoding_size,
    )

    if args.encoder_only:
        return backbone[0].encoder, None, None
    if args.backbone_only:
        return backbone, None, None

    args.num_feature_levels = len(args.projector_scale)

    # Build transformer
    # If using enhanced deformable attention, we would need to modify the transformer
    # For now, we use the standard transformer (you can extend this later)
    transformer = build_transformer(args)

    # Build enhanced segmentation head if segmentation is enabled
    if args.segmentation_head:
        use_color_attention = getattr(args, 'use_color_attention', True)
        use_boundary_refinement = getattr(args, 'use_boundary_refinement', True)

        segmentation_head = EnhancedSegmentationHead(
            in_dim=args.hidden_dim,
            num_blocks=args.dec_layers,
            downsample_ratio=args.mask_downsample_ratio,
            use_color_attention=use_color_attention,
            use_boundary_refinement=use_boundary_refinement
        )
    else:
        segmentation_head = None

    # Build model
    model = LWDETR(
        backbone,
        transformer,
        segmentation_head,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        group_detr=args.group_detr,
        two_stage=args.two_stage,
        lite_refpoint_refine=args.lite_refpoint_refine,
        bbox_reparam=args.bbox_reparam
    )

    # Load pretrained weights if specified
    if args.pretrain_weights:
        try:
            checkpoint = torch.load(args.pretrain_weights, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            # Filter out incompatible keys (from enhanced modules)
            model_state_dict = model.state_dict()
            compatible_state_dict = {
                k: v for k, v in state_dict.items()
                if k in model_state_dict and model_state_dict[k].shape == v.shape
            }

            missing_keys, unexpected_keys = model.load_state_dict(compatible_state_dict, strict=False)
            print(f"Loaded pretrained weights from {args.pretrain_weights}")
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else f"Unexpected keys: {unexpected_keys}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights from {args.pretrain_weights}: {e}")

    model.to(device)

    return model, None, None


def build_enhanced_criterion_and_postprocessors(args):
    """
    Build enhanced criterion with Color Contrast Loss and standard postprocessors.
    """
    device = torch.device(args.device)
    matcher = build_matcher(args)

    # Build weight dictionary
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef
    }

    if args.segmentation_head:
        weight_dict['loss_mask_ce'] = args.mask_ce_loss_coef
        weight_dict['loss_mask_dice'] = args.mask_dice_loss_coef

        # Add color contrast loss weight
        use_color_contrast = getattr(args, 'use_color_contrast_loss', True)
        if use_color_contrast:
            color_contrast_weight = getattr(args, 'color_contrast_loss_weight', 0.5)
            weight_dict['loss_color_contrast'] = color_contrast_weight

    # Handle auxiliary losses
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        if args.two_stage:
            aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # Define losses
    losses = ['labels', 'boxes', 'cardinality']
    if args.segmentation_head:
        losses.append('masks')
        # Add color contrast loss if enabled
        use_color_contrast = getattr(args, 'use_color_contrast_loss', True)
        if use_color_contrast:
            losses.append('color_contrast')

    sum_group_losses = getattr(args, 'sum_group_losses', False)

    # Build enhanced criterion
    if args.segmentation_head:
        use_color_contrast = getattr(args, 'use_color_contrast_loss', True)
        criterion = EnhancedSetCriterion(
            num_classes=args.num_classes + 1,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=args.focal_alpha,
            losses=losses,
            group_detr=args.group_detr,
            sum_group_losses=sum_group_losses,
            use_varifocal_loss=args.use_varifocal_loss,
            use_position_supervised_loss=args.use_position_supervised_loss,
            ia_bce_loss=args.ia_bce_loss,
            mask_point_sample_ratio=args.mask_point_sample_ratio,
            use_color_contrast_loss=use_color_contrast,
            color_contrast_temperature=0.07,
            color_contrast_margin=0.5,
            color_contrast_weight=getattr(args, 'color_contrast_loss_weight', 0.5)
        )
    else:
        # For non-segmentation models, use standard criterion
        from rfdetr.models.lwdetr import SetCriterion
        criterion = SetCriterion(
            num_classes=args.num_classes + 1,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=args.focal_alpha,
            losses=losses,
            group_detr=args.group_detr,
            sum_group_losses=sum_group_losses,
            use_varifocal_loss=args.use_varifocal_loss,
            use_position_supervised_loss=args.use_position_supervised_loss,
            ia_bce_loss=args.ia_bce_loss
        )

    criterion.to(device)
    postprocess = PostProcess(num_select=args.num_select)

    return criterion, postprocess
