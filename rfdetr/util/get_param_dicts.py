# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

import torch.nn as nn

from rfdetr.models.backbone import Joiner


def get_param_dict(args, model_without_ddp: nn.Module):
    assert isinstance(model_without_ddp.backbone, Joiner)
    backbone = model_without_ddp.backbone[0]
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")
    backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]

    decoder_key = 'transformer.decoder'
    decoder_params = [
        p
        for n, p in model_without_ddp.named_parameters() if decoder_key in n and p.requires_grad
    ]

    decoder_param_lr_pairs = [
        {"params": param, "lr": args.lr * args.lr_component_decay} 
        for param in decoder_params
    ]

    other_params = [
        p
        for n, p in model_without_ddp.named_parameters() if (
            n not in backbone_named_param_lr_pairs and decoder_key not in n and p.requires_grad)
    ]
    other_param_dicts = [
        {"params": param, "lr": args.lr} 
        for param in other_params
    ]
    
    final_param_dicts = (
        other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs
    )

    return final_param_dicts
