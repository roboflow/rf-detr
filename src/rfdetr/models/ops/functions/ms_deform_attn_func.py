# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------
"""
ms_deform_attn_func
"""

from __future__ import absolute_import, division, print_function

import torch

from rfdetr.utilities.tensors import _bilinear_grid_sample


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    """ "for debug and test only, need to use cuda version instead"""
    # batch_size, n_heads, head_dim, token_count
    batch_size, n_heads, head_dim, _ = value.shape
    _, query_length, n_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([height * width for height, width in value_spatial_shapes], dim=3)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (height, width) in enumerate(value_spatial_shapes):
        # batch_size, n_heads, head_dim, height, width
        value_l_ = value_list[lid_].view(batch_size * n_heads, head_dim, height, width)
        # batch_size, query_length, n_heads, num_points, 2 -> batch_size, n_heads, query_length, num_points, 2
        # -> batch_size*n_heads, query_length, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # batch_size*n_heads, head_dim, query_length, num_points
        sampling_value_l_ = _bilinear_grid_sample(value_l_, sampling_grid_l_, padding_mode="zeros", align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (batch_size, query_length, n_heads, num_levels * num_points)
    # -> (batch_size, n_heads, query_length, num_levels, num_points)
    # -> (batch_size*n_heads, 1, query_length, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        batch_size * n_heads, 1, query_length, num_levels * num_points
    )
    # batch_size*n_heads, head_dim, query_length, num_levels*num_points
    sampling_value_list = torch.stack(sampling_value_list, dim=-2).flatten(-2)
    output = (sampling_value_list * attention_weights).sum(-1).view(batch_size, n_heads * head_dim, query_length)
    return output.transpose(1, 2).contiguous()
