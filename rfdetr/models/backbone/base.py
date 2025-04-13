# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torch import nn


class BackboneBase(nn.Module):
    def __init__(self):
        super().__init__()

    def get_named_param_lr_pairs(self, args, prefix:str):
        raise NotImplementedError
