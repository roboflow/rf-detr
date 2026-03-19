# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Checkpoint and state-dict helpers."""

import os
import tempfile
from collections import OrderedDict
from typing import Any, Dict, Optional

from rfdetr.utilities.logger import get_logger

logger = get_logger()


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


def validate_checkpoint_compatibility(checkpoint: Dict[str, Any], model_args: Any) -> None:
    """Validate that a checkpoint is compatible with the model configuration.

    Checks for mismatches in ``segmentation_head`` and ``patch_size`` between
    the checkpoint's saved training arguments and the current model configuration.
    Raises a descriptive :class:`ValueError` before ``load_state_dict`` fires so
    that users receive a clear, actionable message instead of a cryptic tensor
    size mismatch error.

    If either side is missing an attribute (e.g. a legacy checkpoint saved before
        ``segmentation_head`` or ``patch_size`` was added to ``args``), that specific
        check is skipped silently — this preserves backwards compatibility with
        pre-existing checkpoints.

    Args:
        checkpoint: Loaded checkpoint dictionary, expected to contain an optional
            ``"args"`` key with training namespace attributes.
        model_args: Namespace (e.g. ``types.SimpleNamespace``) with at least
            ``segmentation_head`` and ``patch_size`` attributes describing the
            current model.

    Raises:
        ValueError: If ``segmentation_head`` or ``patch_size`` in the checkpoint
            args do not match those of the model.

    Note:
        This helper does not mutate ``model_args``. It emits ``logger.warning``
        (not an exception) for class-count mismatches so that callers can still
        proceed with reinitialization or weight loading.

        Two scenarios are distinguished:

        * Backbone pretrain: the checkpoint head was trained with more classes
          than the current ``model_args.num_classes``. In this case the detection
          head is typically reinitialized or trimmed externally to match the
          configured number of classes.
        * Fine-tuned checkpoint: the checkpoint head was trained with fewer
          classes than the current ``model_args.num_classes``. If you intend to
          reuse the checkpoint's classification head as-is, set
          ``model_args.num_classes`` to ``ckpt_num_classes - 1`` (the value
          reported in the warning) before loading the state dict to align the
          configuration and silence the warning.
    """
    # Emit actionable class-count mismatch warning early, before any reinit happens.
    ckpt_class_bias = checkpoint.get("model", {}).get("class_embed.bias", None)
    if ckpt_class_bias is not None:
        ckpt_num_classes = ckpt_class_bias.shape[0]
        model_num_classes: Optional[int] = getattr(model_args, "num_classes", None)
        if model_num_classes is not None and ckpt_num_classes != model_num_classes + 1:
            if model_num_classes + 1 < ckpt_num_classes:
                # Backbone pretrain scenario: checkpoint has more classes, head will be trimmed.
                logger.warning(
                    "Checkpoint has %d classes but model is configured for %d. "
                    "The detection head will be re-initialized to %d classes.",
                    ckpt_num_classes - 1,
                    model_num_classes,
                    model_num_classes,
                )
            else:
                # Fine-tuned checkpoint loaded with wrong (larger) num_classes.
                logger.warning(
                    "Checkpoint has %d classes but model is configured for %d. "
                    "Using checkpoint class count (%d). "
                    "Pass num_classes=%d to suppress this warning.",
                    ckpt_num_classes - 1,
                    model_num_classes,
                    ckpt_num_classes - 1,
                    ckpt_num_classes - 1,
                )

    if "args" not in checkpoint:
        return

    ckpt_args = checkpoint["args"]
    ckpt_segmentation_head: Optional[bool] = getattr(ckpt_args, "segmentation_head", None)
    model_segmentation_head: Optional[bool] = getattr(model_args, "segmentation_head", None)

    if ckpt_segmentation_head is not None and model_segmentation_head is not None:
        if ckpt_segmentation_head != model_segmentation_head:
            if ckpt_segmentation_head:
                raise ValueError(
                    "The checkpoint was trained with a segmentation head, but the current model does not have one. "
                    "Load the weights into a segmentation model (e.g. RFDETRSegNano) instead of a detection model."
                )
            else:
                raise ValueError(
                    "The current model has a segmentation head, but the checkpoint was trained without one. "
                    "Load the weights into a detection model (e.g. RFDETRNano) instead of a segmentation model."
                )

    ckpt_patch_size: Optional[int] = getattr(ckpt_args, "patch_size", None)
    model_patch_size: Optional[int] = getattr(model_args, "patch_size", None)
    if ckpt_patch_size is not None and model_patch_size is not None and ckpt_patch_size != model_patch_size:
        raise ValueError(
            f"The checkpoint was trained with patch_size={ckpt_patch_size}, but the current model uses "
            f"patch_size={model_patch_size}. The checkpoint is incompatible with this model architecture. "
            "To resolve this, either instantiate/configure the model with the checkpoint's patch_size or "
            "use a checkpoint that was trained with the same patch_size as the current model."
        )
