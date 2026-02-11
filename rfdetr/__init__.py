# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from rfdetr.detr import (
    RFDETRBase,
    RFDETRLarge,
    RFDETRLargeDeprecated,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegPreview,
    RFDETRSegSmall,
    RFDETRSegXLarge,
    RFDETRSmall,
)
from rfdetr.platform.models import (
    RFDETR2XLarge,
    RFDETRXLarge,
)


def from_checkpoint(path, **kwargs):
    """Load an RF-DETR model from a training checkpoint, automatically
    inferring the model size from the saved args.

    Args:
        path: Path to a checkpoint file (e.g. ``checkpoint_best_total.pth``).
        **kwargs: Additional keyword arguments forwarded to the model constructor
            (e.g. ``accept_platform_model_license=True`` for XLarge/2XLarge).

    Returns:
        An instance of the appropriate RFDETR model subclass.
    """
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    weights_name = str(args.pretrain_weights).lower()

    # Ordered from most-specific to least-specific so that e.g. "xxlarge"
    # matches before "xlarge", and "xlarge" before "large".
    _MODEL_MAP = [
        ("seg-2xlarge", RFDETRSeg2XLarge),
        ("seg-xlarge", RFDETRSegXLarge),
        ("seg-large", RFDETRSegLarge),
        ("seg-medium", RFDETRSegMedium),
        ("seg-small", RFDETRSegSmall),
        ("seg-nano", RFDETRSegNano),
        ("seg-preview", RFDETRSegPreview),
        ("xxlarge", RFDETR2XLarge),
        ("xlarge", RFDETRXLarge),
        ("large", RFDETRLarge),
        ("medium", RFDETRMedium),
        ("small", RFDETRSmall),
        ("nano", RFDETRNano),
        ("base", RFDETRBase),
    ]

    model_cls = None
    for name, cls in _MODEL_MAP:
        if name in weights_name:
            model_cls = cls
            break

    if model_cls is None:
        raise ValueError(
            f"Could not infer model size from pretrain_weights={args.pretrain_weights!r}. "
            "Please instantiate the model class directly."
        )

    return model_cls(pretrain_weights=path, num_classes=args.num_classes, **kwargs)
