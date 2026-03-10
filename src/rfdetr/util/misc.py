# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""Deprecated: most symbols have moved to ``rfdetr.utilities``.

``accuracy``, ``inverse_sigmoid``, and ``interpolate`` remain here temporarily
and will move to ``rfdetr.models.math`` in Phase 10.
"""

import warnings

warnings.warn(
    "rfdetr.util.misc is deprecated; use rfdetr.utilities instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export symbols that have moved to utilities/.
from rfdetr.utilities.distributed import (  # noqa: E402, F401
    all_gather,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    reduce_dict,
    save_on_master,
)
from rfdetr.utilities.package import get_sha  # noqa: E402, F401
from rfdetr.utilities.state_dict import strip_checkpoint  # noqa: E402, F401
from rfdetr.utilities.tensors import (  # noqa: E402, F401
    NestedTensor,
    _max_by_axis,
    _onnx_nested_tensor_from_tensor_list,
    collate_fn,
    nested_tensor_from_tensor_list,
)

# ---------------------------------------------------------------------------
# accuracy, inverse_sigmoid, interpolate remain here until Phase 10
# when they move to rfdetr.models.math.
# ---------------------------------------------------------------------------

from typing import List, Optional, Tuple

import torch
import torchvision
from torch import Tensor

if float(torchvision.__version__.split(".")[1]) < 7.0:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)) -> List[torch.Tensor]:
    """Computes the precision@k for the specified values of k."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(
    input: Tensor,
    size: Optional[List[int]] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
) -> Tensor:
    """Equivalent to nn.functional.interpolate, but with support for empty batch sizes."""
    if float(torchvision.__version__.split(".")[1]) < 7.0:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(input, size, scale_factor, mode, align_corners)

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


__all__ = [
    # distributed
    "all_gather",
    "get_rank",
    "get_world_size",
    "is_dist_avail_and_initialized",
    "is_main_process",
    "reduce_dict",
    "save_on_master",
    # tensors
    "NestedTensor",
    "collate_fn",
    "nested_tensor_from_tensor_list",
    "_max_by_axis",
    "_onnx_nested_tensor_from_tensor_list",
    # package
    "get_sha",
    # state_dict
    "strip_checkpoint",
    # math (temporarily here, move to models.math in Phase 10)
    "accuracy",
    "interpolate",
    "inverse_sigmoid",
]
