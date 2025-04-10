# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""
CustomOpSymbolicRegistry class
"""
from copy import deepcopy

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args
from torch.onnx.symbolic_helper import _get_tensor_dim_size, _get_tensor_sizes
from torch.autograd import Function


class CustomOpSymbolicRegistry:
    # _SYMBOLICS = {}
    _OPTIMIZER = []

    @classmethod
    def optimizer(cls, fn):
        cls._OPTIMIZER.append(fn)


def register_optimizer():
    def optimizer_wrapper(fn):
        CustomOpSymbolicRegistry.optimizer(fn)
        return fn
    return optimizer_wrapper