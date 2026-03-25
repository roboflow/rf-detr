# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Extracted from lwdetr.py (Phase 10)
# Original copyrights: LW-DETR (Baidu), Conditional DETR (Microsoft),
# DETR (Facebook), Deformable DETR (SenseTime)
# ------------------------------------------------------------------------

"""Post-processing module for converting model outputs to COCO API format."""

import torch
import torch.nn.functional as F
from torch import nn

from rfdetr.utilities import box_ops


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        out_masks = outputs.get("pred_masks", None)
        out_keypoints = outputs.get("pred_keypoints", None)

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # Build results for each image
        results = []
        for i in range(out_logits.shape[0]):
            res_i = {"scores": scores[i], "labels": labels[i], "boxes": boxes[i]}
            k_idx = topk_boxes[i]
            h, w = target_sizes[i].tolist()

            # Optionally gather masks corresponding to the same top-K queries
            if out_masks is not None:
                masks_i = torch.gather(
                    out_masks[i],
                    0,
                    k_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, out_masks.shape[-2], out_masks.shape[-1]),
                )  # [K, Hm, Wm]
                masks_i = F.interpolate(
                    masks_i.unsqueeze(1),
                    size=(int(h), int(w)),
                    mode="bilinear",
                    align_corners=False,
                )  # [K,1,H,W]
                res_i["masks"] = masks_i > 0.0

            # Optionally gather keypoints and scale to pixel coordinates
            if out_keypoints is not None:
                num_keypoints = out_keypoints.shape[2]
                kpts_i = torch.gather(
                    out_keypoints[i],
                    0,
                    k_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, num_keypoints, 3),
                )  # [K, num_keypoints, 3]

                # Scale coordinates from [0,1] to pixel space
                kpts_i_scaled = kpts_i.clone()
                kpts_i_scaled[..., 0] = kpts_i[..., 0] * w  # x
                kpts_i_scaled[..., 1] = kpts_i[..., 1] * h  # y
                # Convert visibility logits to confidence scores via sigmoid
                vis_conf = kpts_i[..., 2].sigmoid()
                # For COCO evaluation: 2 = visible, 0 = not labeled
                kpts_i_scaled[..., 2] = (vis_conf > 0.5).float() * 2

                res_i["keypoints"] = kpts_i_scaled
                # Also store raw visibility confidence for user access
                res_i["keypoints_confidence"] = vis_conf

            results.append(res_i)

        return results
