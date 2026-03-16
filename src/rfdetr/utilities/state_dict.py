# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Checkpoint and state-dict helpers."""

import os
import tempfile
from collections import OrderedDict
from typing import Any, Dict


def strip_checkpoint(checkpoint: str | os.PathLike[str]) -> None:
    """Strip a checkpoint file down to ``model`` and ``args`` keys only.

    Overwrites the file atomically so a partial write cannot corrupt it.

    Args:
        checkpoint: Path to the ``.pth`` checkpoint file to strip in place.
    """
    import torch

    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    new_state_dict = {
        "model": state_dict["model"],
        "args": state_dict["args"],
    }
    # Create the temp file in the destination directory so os.replace stays on the same filesystem (atomic).
    checkpoint_dir = os.path.dirname(os.path.abspath(os.fspath(checkpoint)))
    with tempfile.NamedTemporaryFile(dir=checkpoint_dir, delete=False) as tmp_file:
        tmp_path = tmp_file.name
    try:
        torch.save(new_state_dict, tmp_path)
        # Atomic replace avoids leaving a partially written checkpoint on save failures/interruption.
        os.replace(tmp_path, checkpoint)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def clean_state_dict(state_dict: Dict[str, Any]) -> OrderedDict[str, Any]:
    """Remove the ``module.`` prefix added by ``DataParallel`` / ``DistributedDataParallel``.

    Args:
        state_dict: State dict potentially containing ``module.``-prefixed keys.

    Returns:
        New ``OrderedDict`` with ``module.`` stripped from all keys.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict
