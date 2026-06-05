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
import torch.nn.functional as F  # noqa: N812
from torch import nn

from rfdetr.utilities import box_ops


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api."""

    def __init__(
        self,
        num_select: int = 300,
        num_keypoints_per_class: list[int] | None = None,
        trace_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_select = num_select
        self.num_keypoints_per_class = num_keypoints_per_class or []
        self.trace_alpha = trace_alpha

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation) For
                          visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        out_masks = outputs.get("pred_masks", None)
        out_keypoints = outputs.get("pred_keypoints", None)

        assert not (out_masks is not None and out_keypoints is not None), (
            "masks and keypoints cannot be used together in postprocessing."
        )

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        logits_for_topk = prob.view(out_logits.shape[0], -1)
        num_to_select = min(self.num_select, logits_for_topk.shape[1])
        topk_values, topk_indexes = torch.topk(logits_for_topk, num_to_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # Optionally gather masks corresponding to the same top-K queries and resize to original size
        results = []
        if out_masks is not None:
            for i in range(out_masks.shape[0]):
                res_i = {"scores": scores[i], "labels": labels[i], "boxes": boxes[i]}
                k_idx = topk_boxes[i]
                masks_i = torch.gather(
                    out_masks[i],
                    0,
                    k_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, out_masks.shape[-2], out_masks.shape[-1]),
                )  # [K, Hm, Wm]
                h, w = target_sizes[i].tolist()
                masks_i = F.interpolate(
                    masks_i.unsqueeze(1),
                    size=(int(h), int(w)),
                    mode="bilinear",
                    align_corners=False,
                )  # [K,1,H,W]
                res_i["masks"] = masks_i > 0.0
                results.append(res_i)
        elif out_keypoints is not None:
            max_num_keypoints = max(self.num_keypoints_per_class, default=0)
            num_keypoint_classes = len(self.num_keypoints_per_class)
            for i in range(out_keypoints.shape[0]):
                labels_i = labels[i]
                boxes_i = boxes[i]
                scores_i = scores[i]
                keypoint_query_indices = topk_boxes[i]
                keypoints_i = torch.gather(
                    out_keypoints[i],
                    0,
                    keypoint_query_indices.unsqueeze(-1)
                    .unsqueeze(-1)
                    .repeat(
                        1,
                        out_keypoints.shape[-2],
                        out_keypoints.shape[-1],
                    ),
                )

                output_keypoints = keypoints_i.new_zeros((keypoints_i.shape[0], max_num_keypoints, 3))
                output_keypoint_precision = keypoints_i.new_full(
                    (keypoints_i.shape[0], max_num_keypoints, 3), float("nan")
                )
                if num_keypoint_classes > 0 and max_num_keypoints > 0:
                    reshaped = keypoints_i.view(
                        keypoints_i.shape[0], num_keypoint_classes, max_num_keypoints, keypoints_i.shape[-1]
                    )
                    valid_class_mask = labels_i < num_keypoint_classes
                    if valid_class_mask.any():
                        valid_indices = valid_class_mask.nonzero(as_tuple=True)[0]
                        selected_labels = labels_i[valid_indices]
                        selected_keypoints = reshaped[valid_indices, selected_labels]
                        if self.trace_alpha > 0 and selected_keypoints.shape[-1] >= 7:
                            log_mean_traces = []
                            for selected_pos, selected_label_tensor in enumerate(selected_labels):
                                selected_label = int(selected_label_tensor.item())
                                num_active_keypoints = self.num_keypoints_per_class[selected_label]
                                if num_active_keypoints <= 0:
                                    log_mean_traces.append(selected_keypoints.new_tensor(0.0))
                                    continue

                                active_keypoints = selected_keypoints[selected_pos, :num_active_keypoints]
                                log_l11 = active_keypoints[:, 4]
                                l21 = active_keypoints[:, 5]
                                log_l22 = active_keypoints[:, 6]
                                w_find = active_keypoints[:, 2].sigmoid()
                                log_t1 = -2.0 * log_l11
                                log_t2 = -2.0 * log_l22
                                log_t3 = 2.0 * torch.log(l21.abs().clamp(min=1e-12)) + log_t1 + log_t2
                                log_trace_sigma = torch.logsumexp(torch.stack([log_t1, log_t2, log_t3], dim=-1), dim=-1)
                                log_w_find = torch.log(w_find.clamp(min=1e-12))
                                log_mean_trace = torch.logsumexp(
                                    log_trace_sigma + log_w_find, dim=-1
                                ) - torch.logsumexp(log_w_find, dim=-1)
                                log_mean_traces.append(log_mean_trace)

                            scores_i = scores_i.clone()
                            scores_i[valid_indices] = scores_i[valid_indices] * torch.exp(
                                -self.trace_alpha * torch.stack(log_mean_traces)
                            )
                        img_h, img_w = target_sizes[i]
                        for selected_pos, output_index in enumerate(valid_indices):
                            selected_label = int(selected_labels[selected_pos].item())
                            num_active_keypoints = self.num_keypoints_per_class[selected_label]
                            if num_active_keypoints <= 0:
                                continue

                            active_keypoints = selected_keypoints[selected_pos, :num_active_keypoints]
                            output_keypoints[output_index, :num_active_keypoints, 0] = active_keypoints[:, 0] * img_w
                            output_keypoints[output_index, :num_active_keypoints, 1] = active_keypoints[:, 1] * img_h
                            output_keypoints[output_index, :num_active_keypoints, 2] = active_keypoints[:, 2].sigmoid()
                            if active_keypoints.shape[-1] >= 7:
                                output_keypoint_precision[output_index, :num_active_keypoints] = active_keypoints[
                                    :, 4:7
                                ]

                results.append(
                    {
                        "scores": scores_i,
                        "labels": labels_i,
                        "boxes": boxes_i,
                        "keypoints": output_keypoints,
                        "keypoint_precision_cholesky": output_keypoint_precision,
                    }
                )
        else:
            results = [
                {"scores": score, "labels": label, "boxes": box} for score, label, box in zip(scores, labels, boxes)
            ]

        return results
