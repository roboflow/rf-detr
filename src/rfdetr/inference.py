# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ModelContext and model-context builder for RF-DETR inference."""

from __future__ import annotations

__all__ = ["ModelContext"]

from typing import TYPE_CHECKING, Any, Callable, List, Optional, cast

import torch

from rfdetr.config import TrainConfig
from rfdetr.models import PostProcess, build_model
from rfdetr.models.weights import apply_lora, load_pretrain_weights

if TYPE_CHECKING:
    from rfdetr.config import ModelConfig


class ModelContext:
    """Lightweight model wrapper returned by RFDETR.get_model().

    Provides the same attribute interface as the legacy ``main.py:Model`` but
    without importing or depending on ``populate_args()`` or the legacy stack.

    Args:
        model: The underlying ``nn.Module`` (LWDETR instance).
        postprocess: PostProcess instance for converting raw outputs to boxes.
        device: Device the model lives on.
        resolution: Input resolution (square side length in pixels).
        args: Namespace produced by :func:`build_namespace`.
        class_names: Optional list of class name strings loaded from checkpoint.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        postprocess: PostProcess,
        device: torch.device,
        resolution: int,
        args: Any,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.postprocess = postprocess
        self.device = device
        self.resolution = resolution
        self.args = args
        self.class_names = class_names
        self.inference_model = None

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Reinitialize the detection head for a different number of classes.

        Args:
            num_classes: New number of output classes (including background).
        """
        reinitialize_head = cast(Callable[[int], None], getattr(self.model, "reinitialize_detection_head"))
        reinitialize_head(num_classes)
        self.args.num_classes = num_classes


_ModelContext = ModelContext  # backward-compat alias


def _build_model_context(model_config: ModelConfig) -> ModelContext:
    """Build a ModelContext from ModelConfig without using legacy main.py:Model.

    Replicates ``Model.__init__`` logic: builds the nn.Module, optionally loads
    pretrain weights and applies LoRA, then moves the model to the target device.

    Args:
        model_config: Architecture configuration.

    Returns:
        Fully initialised ModelContext ready for inference or training.
    """
    from rfdetr._namespace import _namespace_from_configs

    # A dummy TrainConfig is needed only for _namespace_from_configs' required fields;
    # dataset_dir/output_dir are unused during model construction.
    dummy_train_config = TrainConfig(dataset_dir=".", output_dir=".")
    args = _namespace_from_configs(model_config, dummy_train_config)
    nn_model = build_model(args)

    class_names: List[str] = []
    if model_config.pretrain_weights is not None:
        class_names = load_pretrain_weights(nn_model, model_config)
        # ``load_pretrain_weights`` can mutate ``model_config.num_classes`` when
        # aligning to checkpoint heads. Keep the derived namespace in sync.
        if hasattr(args, "num_classes") and getattr(args, "num_classes") != model_config.num_classes:
            args.num_classes = model_config.num_classes

    if model_config.backbone_lora:
        apply_lora(nn_model)

    device = torch.device(args.device)
    nn_model = nn_model.to(device)
    postprocess = PostProcess(num_select=args.num_select)

    return ModelContext(
        model=nn_model,
        postprocess=postprocess,
        device=device,
        resolution=model_config.resolution,
        args=args,
        class_names=class_names or None,
    )
