# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from pydantic import BaseModel, field_validator, model_validator, Field
from pydantic_core.core_schema import ValidationInfo  # for field_validator(info)
from typing import List, Optional, Literal
import os, torch

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# centralize all supported encoder names (add dinov3).
EncoderName = Literal[
    "dinov2_windowed_small",
    "dinov2_windowed_base",
    "dinov3_small",
    "dinov3_base",
    "dinov3_large",
]

def _encoder_default():
    """Default encoder name for the model config."""
    # default to v2 unless explicitly overridden by env
    val = os.getenv("RFD_ENCODER", "").strip() or "dinov2_windowed_small"

    # guardrail: only accept known names
    allowed = {
        "dinov2_windowed_small","dinov2_windowed_base",
        "dinov3_small","dinov3_base","dinov3_large"
    }
    return val if val in allowed else "dinov2_windowed_small"

class ModelConfig(BaseModel):
    """Base configuration for RF-DETR models."""
    # WAS: only dinov2_windowed_*; NOW: include dinov3_* as drop-in options
    encoder: EncoderName = _encoder_default()

    out_feature_indexes: List[int]
    dec_layers: int
    two_stage: bool = True
    projector_scale: List[Literal["P3", "P4", "P5"]]
    hidden_dim: int
    patch_size: int
    num_windows: int
    sa_nheads: int
    ca_nheads: int
    dec_n_points: int
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_classes: int = 90
    pretrain_weights: Optional[str] = None
    device: Literal["cpu", "cuda", "mps"] = DEVICE
    resolution: int
    group_detr: int = 13
    gradient_checkpointing: bool = False
    positional_encoding_size: int
    # used only when encoder startswith("dinov3")
    dinov3_repo_dir: Optional[str] = None     # e.g., r"D:\repos\dinov3"
    dinov3_weights_path: Optional[str] = None # e.g., r"C:\models\dinov3-vitb16.pth"
    dinov3_hf_token: Optional[str] = None     # or rely on HUGGINGFACE_HUB_TOKEN
    dinov3_prefer_hf: bool = True             # try HF first, then hub fallback

    # force /16 for v3
    @field_validator("patch_size", mode="after")
    def _coerce_patch_for_dinov3(cls, v, info: ValidationInfo):
        """Ensure patch size is 16 for DINOv3 encoders."""
        enc = str(info.data.get("encoder", ""))
        return 16 if enc.startswith("dinov3") else v
    
    # keep pos-encoding grid consistent with resolution / patch
    @field_validator("positional_encoding_size", mode="after")
    def _sync_pos_enc_with_resolution(cls, v, info: ValidationInfo):
        """Sync positional encoding size with resolution and patch size."""
        values = info.data or {}
        res, ps = values.get("resolution"), values.get("patch_size")
        return max(1, res // ps) if (res and ps) else v

    # env fallbacks for local repo/weights when *not* preferring HF
    @field_validator("dinov3_repo_dir", "dinov3_weights_path", mode="after")
    def _fallback_to_env(cls, v, info: ValidationInfo):
        """Fallback to environment variables if not set."""
        values = info.data or {}
        if (not v) and str(values.get("encoder","")).startswith("dinov3") and not values.get("dinov3_prefer_hf", True):
            env_map = {"dinov3_repo_dir": "DINOV3_REPO", "dinov3_weights_path": "DINOV3_WEIGHTS"}
            env_key = env_map[info.field_name]
            return os.getenv(env_key, v)
        return v

    # neutralize windowing for v3 (avoid accidental asserts downstream)
    @field_validator("num_windows", mode="after")
    def _neutralize_windows_for_dinov3(cls, v, info: ValidationInfo):
        """Neutralize windowing for DINOv3 encoders."""
        enc = str((info.data or {}).get("encoder",""))
        return 1 if enc.startswith("dinov3") else v
    
    # auto-fit out_feature_indexes to avoid projector shape mismatches
    @field_validator("out_feature_indexes", mode="after")
    def _coerce_out_feats_for_backbone(cls, v, info: ValidationInfo):
        """Ensure out_feature_indexes are compatible with the encoder."""
        enc = str((info.data or {}).get("encoder",""))
        if enc.startswith("dinov3"):
            # DINOv3 path: default to fewer, stable high-level features
            return v if len(v) in (2,) else [8, 11]
        return v
    
    # Final safety net: once the whole model is built, normalize settings for DINOv3.
    @model_validator(mode="after")
    def _final_autofix_for_dinov3(self):
        """Final adjustments after model construction."""
        enc = str(self.encoder)
        if enc.startswith("dinov3"):
            # enforce /16 patch + matching pos-enc grid
            self.patch_size = 16
            if self.resolution:
                self.positional_encoding_size = max(1, self.resolution // self.patch_size)
            # windowing is a no-op for v3
            self.num_windows = 1
            # most important: use 2 high-level features to match projector weights across v2/v3
            if len(self.out_feature_indexes) != 2:
                self.out_feature_indexes = [8, 11]
        return self
    
class RFDETRBaseConfig(ModelConfig):
    """
    The configuration for an RF-DETR Base model.
    """
    # Allow choosing dinov3_* without changing call sites
    encoder: EncoderName = _encoder_default()
    print("Using RFDETRBaseConfig with encoder:", encoder)
    hidden_dim: int = 256
    patch_size: int = 14           # will auto-become 16 if encoder startswith("dinov3")
    num_windows: int = 4           # ignored by DINOv3 branch
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P4"]
    out_feature_indexes: List[int] = [2, 4, 5, 9]
    pretrain_weights: Optional[str] = "rf-detr-base.pth"
    #resolution: int = 504          # 560//16=35 when dinov3_* is used
    resolution: int = 512          # 512//16=32 → pos grid auto=32 for both v2/v3
    positional_encoding_size: int = 36  # will auto-sync to resolution//patch_size


class RFDETRLargeConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Large model.
    """
    encoder: EncoderName = "dinov2_windowed_base"
    hidden_dim: int = 384
    sa_nheads: int = 12
    ca_nheads: int = 24
    dec_n_points: int = 4
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P3", "P5"]
    pretrain_weights: Optional[str] = "rf-detr-large.pth"


class RFDETRNanoConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Nano model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 2
    patch_size: int = 16
    resolution: int = 384          # 384//16=24 → pos grid auto=24 for both v2/v3
    positional_encoding_size: int = 24
    pretrain_weights: Optional[str] = "rf-detr-nano.pth"


class RFDETRSmallConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Small model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 3
    patch_size: int = 16
    resolution: int = 512          # 512//16=32 → pos grid auto=32
    positional_encoding_size: int = 32
    pretrain_weights: Optional[str] = "rf-detr-small.pth"


class RFDETRMediumConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Medium model.
    """
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 16
    #resolution: int = 504          # 576//16=36 → pos grid auto=36
    resolution: int = 512
    positional_encoding_size: int = 36
    pretrain_weights: Optional[str] = "rf-detr-medium.pth"


class TrainConfig(BaseModel):
    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 100
    ema_decay: float = 0.993
    ema_tau: int = 100
    lr_drop: int = 100
    checkpoint_interval: int = 10
    warmup_epochs: int = 0
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    drop_path: float = 0.0
    group_detr: int = 13
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    num_select: int = 300
    dataset_file: Literal["coco", "o365", "roboflow"] = "roboflow"
    square_resize_div_64: bool = True
    dataset_dir: str
    output_dir: str = "output"
    multi_scale: bool = True
    expanded_scales: bool = True
    do_random_resize_via_padding: bool = False
    use_ema: bool = True
    num_workers: int = 2
    weight_decay: float = 1e-4
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = False
    tensorboard: bool = True
    wandb: bool = False
    project: Optional[str] = None
    run: Optional[str] = None
    class_names: List[str] = None
    run_test: bool = True
