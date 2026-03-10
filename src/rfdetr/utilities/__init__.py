# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Utility functions and helpers."""

from rfdetr.utilities import box_ops
from rfdetr.utilities.distributed import (
    all_gather,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    reduce_dict,
    save_on_master,
)
from rfdetr.utilities.files import _compute_file_md5, _download_file, _validate_file_md5
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.package import get_sha, get_version
from rfdetr.utilities.reproducibility import seed_all
from rfdetr.utilities.state_dict import clean_state_dict, strip_checkpoint
from rfdetr.utilities.tensors import (
    NestedTensor,
    _max_by_axis,
    _onnx_nested_tensor_from_tensor_list,
    collate_fn,
    nested_tensor_from_tensor_list,
)

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
    # box_ops (submodule)
    "box_ops",
    # logger
    "get_logger",
    # files
    "_compute_file_md5",
    "_download_file",
    "_validate_file_md5",
    # package
    "get_sha",
    "get_version",
    # reproducibility
    "seed_all",
    # state_dict
    "clean_state_dict",
    "strip_checkpoint",
]
