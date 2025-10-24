"""
Enhanced SetCriterion with Color Contrast Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from rfdetr.models.lwdetr import SetCriterion
from rfdetr.models.enhancements import ColorContrastLoss


class EnhancedSetCriterion(SetCriterion):
    """
    Enhanced criterion that adds Color Contrast Loss to the standard losses.
    """
    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        focal_alpha,
        losses,
        group_detr=1,
        sum_group_losses=False,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        ia_bce_loss=False,
        mask_point_sample_ratio: int = 16,
        use_color_contrast_loss: bool = True,
        color_contrast_temperature: float = 0.07,
        color_contrast_margin: float = 0.5,
        color_contrast_weight: float = 0.5,
    ):
        """
        Args:
            ... (same as SetCriterion)
            use_color_contrast_loss: Whether to use color contrast loss
            color_contrast_temperature: Temperature for contrastive loss
            color_contrast_margin: Margin for pushing instances apart
            color_contrast_weight: Weight for color contrast loss
        """
        super().__init__(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=focal_alpha,
            losses=losses,
            group_detr=group_detr,
            sum_group_losses=sum_group_losses,
            use_varifocal_loss=use_varifocal_loss,
            use_position_supervised_loss=use_position_supervised_loss,
            ia_bce_loss=ia_bce_loss,
            mask_point_sample_ratio=mask_point_sample_ratio,
        )

        self.use_color_contrast_loss = use_color_contrast_loss

        if use_color_contrast_loss:
            self.color_contrast_loss = ColorContrastLoss(
                temperature=color_contrast_temperature,
                margin=color_contrast_margin,
                weight=color_contrast_weight
            )

    def loss_color_contrast(
        self,
        outputs,
        targets,
        indices,
        num_boxes,
        images: Optional[torch.Tensor] = None
    ):
        """
        Compute color contrast loss to encourage different instances to have distinct colors.

        Args:
            outputs: Model outputs containing 'pred_masks'
            targets: Ground truth targets
            indices: Matching indices
            num_boxes: Number of boxes for normalization
            images: RGB images [B, 3, H, W]
        """
        if not self.use_color_contrast_loss or images is None:
            return {'loss_color_contrast': torch.tensor(0.0, device=outputs['pred_masks'].device)}

        assert 'pred_masks' in outputs, "pred_masks missing in model outputs"

        pred_masks = outputs['pred_masks']  # [B, Q, H, W]
        B, Q, H, W = pred_masks.shape

        # Process each batch separately
        total_loss = 0.0
        valid_batches = 0

        for batch_idx in range(B):
            # Get matched indices for this batch
            src_idx, tgt_idx = indices[batch_idx]

            if len(src_idx) <= 1:
                # Need at least 2 instances for contrast
                continue

            # Get predicted masks for matched instances
            batch_pred_masks = pred_masks[batch_idx, src_idx]  # [N, H, W]
            batch_pred_masks = batch_pred_masks.sigmoid()  # Apply sigmoid

            # Get target masks
            target = targets[batch_idx]
            target_masks = target['masks'][tgt_idx]  # [N, Ht, Wt]

            # Resize target masks to match prediction size if needed
            if target_masks.shape[-2:] != (H, W):
                target_masks = F.interpolate(
                    target_masks.unsqueeze(1).float(),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)

            # Get image for this batch
            batch_image = images[batch_idx:batch_idx+1]  # [1, 3, H_img, W_img]

            # Expand dimensions for the loss function
            batch_pred_masks_expanded = batch_pred_masks.unsqueeze(0)  # [1, N, H, W]
            target_masks_expanded = target_masks.unsqueeze(0)  # [1, N, H, W]

            # Compute color contrast loss
            loss = self.color_contrast_loss(
                pred_masks=batch_pred_masks_expanded,
                target_masks=target_masks_expanded,
                images=batch_image
            )

            total_loss += loss
            valid_batches += 1

        # Average over valid batches
        if valid_batches > 0:
            loss_value = total_loss / valid_batches
        else:
            loss_value = torch.tensor(0.0, device=pred_masks.device)

        return {'loss_color_contrast': loss_value}

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """
        Extended version that includes color contrast loss.
        """
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'color_contrast': self.loss_color_contrast,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, images: Optional[torch.Tensor] = None):
        """
        Extended forward that includes images for color contrast loss.

        Args:
            outputs: Model outputs
            targets: Ground truth targets
            images: RGB images [B, 3, H, W] (required for color contrast loss)
        """
        group_detr = self.group_detr if self.training else 1
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_detr=group_detr)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        if not self.sum_group_losses:
            num_boxes = num_boxes * group_detr
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device
        )

        from rfdetr.util.misc import is_dist_avail_and_initialized, get_world_size
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            if loss == 'color_contrast':
                kwargs['images'] = images
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, group_detr=group_detr)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    elif loss == 'color_contrast':
                        kwargs['images'] = images
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            indices = self.matcher(enc_outputs, targets, group_detr=group_detr)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                elif loss == 'color_contrast':
                    kwargs['images'] = images
                l_dict = self.get_loss(loss, enc_outputs, targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses
