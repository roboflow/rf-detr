# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------

from typing import Any, get_args

from torch import Tensor, nn

from rfdetr.config import EncoderName
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.position_encoding import build_position_encoding
from rfdetr.utilities.tensors import NestedTensor


class Joiner(nn.Sequential):
    def __init__(self, backbone: Backbone, position_embedding: nn.Module) -> None:
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor) -> tuple[list[NestedTensor], list[Tensor], list[NestedTensor] | None]:
        """"""
        result = self[0](tensor_list)
        if isinstance(result, tuple):
            x, cross_attn_x = result
        else:
            x, cross_attn_x = result, None
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos, cross_attn_x

    def export(self) -> None:
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export  # type: ignore[method-assign,assignment]
        for name, m in self.named_modules():
            if hasattr(m, "export") and callable(m.export) and hasattr(m, "_export") and not m._export:
                m.export()

    def forward_export(self, inputs: Tensor) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor] | None]:
        result = self[0](inputs)
        if len(result) == 3:
            feats, masks, cross_attn_feats = result
        else:
            feats, masks = result
            cross_attn_feats = None
        poss = []
        for feat, mask in zip(feats, masks):
            pos = self[1](mask, align_dim_orders=False).to(feat.dtype)
            if cross_attn_feats is None and pos.ndim == 4 and pos.shape[1] == 1:
                pos = pos[:, 0]
            poss.append(pos)
        return feats, masks, poss, cross_attn_feats


#: Backbone classes for encoders that are not built into ``rfdetr``, keyed by ``ModelConfig.encoder``.
_BACKBONE_REGISTRY: dict[str, type[Backbone]] = {}


def register_backbone(encoder: str, backbone_cls: type[Backbone]) -> None:
    """Route :func:`build_backbone` for *encoder* to *backbone_cls*.

    Extension packages (for example ``rfdetr_plus``) call this at import time to plug in an encoder that ``rfdetr``
    does not ship. *backbone_cls* subclasses :class:`Backbone` and overrides ``_build_encoder`` (and, when its
    parameters follow other naming rules, ``get_named_param_lr_pairs``); the projector, padding masks and export path
    stay those of :class:`Backbone`. An override of ``_build_encoder`` must accept ``**kwargs`` and use only the
    arguments it needs, because the hook signature may grow.

    Registry keys are ``ModelConfig.encoder`` values, which ``ModelConfig`` restricts to the three built-in DINOv2
    names (``EncoderName``). To select a registered encoder through a config, subclass ``ModelConfig`` (or one of its
    subclasses) and redeclare ``encoder`` with the registered name. A ``Literal`` redeclaration that is not a subtype of
    the base ``Literal`` needs ``# type: ignore[assignment]``.

    The stochastic-depth (drop-path) schedule finds the transformer layers of the module that ``_build_encoder``
    returns at its ``blocks``, ``trunk.blocks`` or ``encoder.encoder.layer`` attribute (the HuggingFace DINOv2 layout),
    and sets ``drop_path.drop_prob`` on each layer that has it. An encoder with another layout is trained without the
    schedule, and a warning is logged when a positive ``drop_path`` is configured.

    Args:
        encoder: The ``ModelConfig.encoder`` value that selects *backbone_cls*.
        backbone_cls: The :class:`Backbone` subclass to build for *encoder*.

    Raises:
        TypeError: If *backbone_cls* is not a :class:`Backbone` subclass.
        ValueError: If *encoder* is a DINOv2 encoder name, or is already registered to another class.
    """
    if not (isinstance(backbone_cls, type) and issubclass(backbone_cls, Backbone)):
        raise TypeError(f"backbone_cls must be a Backbone subclass, got {backbone_cls!r}.")
    if encoder in get_args(EncoderName) or encoder.split("_")[0] == "dinov2":
        raise ValueError(f"Encoder {encoder!r} is a built-in rfdetr (DINOv2) encoder name and cannot be registered.")
    registered = _BACKBONE_REGISTRY.get(encoder)
    if registered is not None and registered is not backbone_cls:
        raise ValueError(
            f"Encoder {encoder!r} is already registered to {registered.__module__}.{registered.__qualname__}."
        )
    _BACKBONE_REGISTRY[encoder] = backbone_cls


def build_backbone(
    encoder: str,
    vit_encoder_num_layers: int,
    pretrained_encoder: str | None,
    window_block_indexes: list[Any] | None,
    drop_path: float,
    out_channels: int,
    out_feature_indexes: list[Any] | None,
    projector_scale: list[Any] | None,
    use_cls_token: bool,
    hidden_dim: int,
    position_embedding: str,
    freeze_encoder: bool,
    layer_norm: bool,
    target_shape: tuple[int, int],
    rms_norm: bool,
    backbone_lora: bool,
    force_no_pretrain: bool,
    gradient_checkpointing: bool,
    load_dinov2_weights: bool,
    patch_size: int,
    num_windows: int,
    positional_encoding_size: int,
    dual_projector: bool = False,
) -> Joiner:
    """
    Useful args:
        - encoder: encoder name
        - lr_encoder:
        - dilation
        - use_checkpoint: for swin only for now

    """
    position_embedding_module = build_position_encoding(hidden_dim, position_embedding)

    backbone_cls = _BACKBONE_REGISTRY.get(encoder, Backbone)
    backbone = backbone_cls(
        encoder,
        pretrained_encoder,
        window_block_indexes=window_block_indexes,
        drop_path=drop_path,
        out_channels=out_channels,
        out_feature_indexes=out_feature_indexes,
        projector_scale=projector_scale,
        use_cls_token=use_cls_token,
        layer_norm=layer_norm,
        freeze_encoder=freeze_encoder,
        target_shape=target_shape,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        gradient_checkpointing=gradient_checkpointing,
        load_dinov2_weights=load_dinov2_weights and not force_no_pretrain,
        patch_size=patch_size,
        num_windows=num_windows,
        positional_encoding_size=positional_encoding_size,
        dual_projector=dual_projector,
    )

    model = Joiner(backbone, position_embedding_module)
    return model
