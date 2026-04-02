# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import warnings
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseConfig(BaseModel):
    """
    Base configuration class that validates input parameters against the defined model schema.
    If any unknown fields are provided, a ValueError is raised listing the unknown and available parameters.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", validate_assignment=True)

    @model_validator(mode="before")
    @classmethod
    def catch_typo_kwargs(cls, values: Any) -> Any:
        if not isinstance(values, Mapping):
            return values
        allowed_params = set(cls.model_fields.keys())
        provided_params = set(values)
        unknown_params = provided_params - allowed_params
        if unknown_params:
            unknown_params_list = ", ".join(f"'{param}'" for param in sorted(unknown_params))
            allowed_params_list = ", ".join(sorted(allowed_params))
            raise ValueError(
                f"Unknown parameter(s): {unknown_params_list}. Available parameter(s): {allowed_params_list}."
            )
        return values

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or name in type(self).model_fields:
            super().__setattr__(name, value)
            return
        raise ValueError(f"Unknown attribute: '{name}'.")


class ModelConfig(BaseConfig):
    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"]
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
    num_queries: int = 300
    # NOTE:
    # - ModelConfig is the authoritative source of `num_select` for PTL/inference; it is read via `build_namespace`.
    # - Any `num_select` field on TrainConfig / SegmentationTrainConfig is deprecated and ignored by PTL/inference.
    num_select: int = 300
    bbox_reparam: bool = True
    lite_refpoint_refine: bool = True
    layer_norm: bool = True
    amp: bool = True
    num_classes: int = 90
    pretrain_weights: Optional[str] = None
    # torch.device values are accepted at validation time and normalized to string.
    device: str = DEVICE
    resolution: int
    group_detr: int = 13
    gradient_checkpointing: bool = False
    compile: bool = False
    fused_optimizer: bool = True
    positional_encoding_size: int
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    segmentation_head: bool = False
    mask_downsample_ratio: int = 4
    backbone_lora: bool = False
    freeze_encoder: bool = False
    license: str = "Apache-2.0"
    model_name: Optional[str] = Field(
        default=None,
        description=(
            'Name of the model class stored in training checkpoints (e.g. ``"RFDETRLarge"``). '
            "Set automatically by ``RFDETR.train()`` before saving. "
            "Used by ``RFDETR.from_checkpoint()`` to resolve the correct subclass directly "
            "without inspecting ``pretrain_weights``."
        ),
    )

    @model_validator(mode="after")
    def _warn_deprecated_model_config_fields(self) -> "ModelConfig":
        """Emit DeprecationWarning when cls_loss_coef is explicitly set on ModelConfig.

        ``cls_loss_coef`` ownership is moving to ``TrainConfig`` (Item #3, v1.7). Setting
        it on ``ModelConfig`` is deprecated.  Use ``TrainConfig(cls_loss_coef=...)`` instead.
        """
        if "cls_loss_coef" in self.model_fields_set:
            # stacklevel=2 points into Pydantic internals rather than the user call
            # site — this is unavoidable with @model_validator(mode="after") in
            # Pydantic v2.  The warning still fires correctly; the origin frame is
            # less precise than ideal.
            warnings.warn(
                "ModelConfig.cls_loss_coef is deprecated and will be removed in v1.9. "
                "Set cls_loss_coef on TrainConfig instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self

    @field_validator("pretrain_weights", mode="after")
    @classmethod
    def expand_path(cls, v: Optional[str]) -> Optional[str]:
        """
        Expand user paths (e.g., '~' or paths with separators) but leave simple filenames
        (like 'rf-detr-base.pth') unchanged so they can match hosted model keys.
        """
        if v is None:
            return v
        return os.path.realpath(os.path.expanduser(v))

    @field_validator("device", mode="before")
    @classmethod
    def _normalize_device(cls, v: Any) -> str:
        """Normalize supported device inputs to a canonical torch-style string.

        Args:
            v: Device specifier provided by callers. Supported values are
                ``str`` (for example ``"cpu"``, ``"cuda"``, ``"cuda:1"``)
                and ``torch.device``.

        Returns:
            Canonical string form of the parsed device (for example ``"cuda:1"``).

        Raises:
            ValueError: If a string value cannot be parsed as a valid torch device.
            ValueError: If ``v`` is not a string or ``torch.device``.
        """
        if isinstance(v, torch.device):
            return str(v)
        if isinstance(v, str):
            try:
                return str(torch.device(v))
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(f"Invalid device specifier: {v!r}.") from exc
        raise ValueError("device must be a string or torch.device.")


class RFDETRBaseConfig(ModelConfig):
    """
    The configuration for an RF-DETR Base model.
    """

    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = "dinov2_windowed_small"
    hidden_dim: int = 256
    patch_size: int = 14
    num_windows: int = 4
    dec_layers: int = 3
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_queries: int = 300
    num_select: int = 300
    projector_scale: List[Literal["P3", "P4", "P5"]] = ["P4"]
    out_feature_indexes: List[int] = [2, 5, 8, 11]
    pretrain_weights: Optional[str] = "rf-detr-base.pth"
    resolution: int = 560
    positional_encoding_size: int = 37


class RFDETRLargeDeprecatedConfig(RFDETRBaseConfig):
    """
    The configuration for an RF-DETR Large model.
    """

    encoder: Literal["dinov2_windowed_small", "dinov2_windowed_base"] = "dinov2_windowed_base"
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
    resolution: int = 384
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
    resolution: int = 512
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
    resolution: int = 576
    positional_encoding_size: int = 36
    pretrain_weights: Optional[str] = "rf-detr-medium.pth"


# res 704, ps 16, 2 windows, 4 dec layers, 300 queries, ViT-S basis
class RFDETRLargeConfig(ModelConfig):
    encoder: Literal["dinov2_windowed_small"] = "dinov2_windowed_small"
    hidden_dim: int = 256
    dec_layers: int = 4
    sa_nheads: int = 8
    ca_nheads: int = 16
    dec_n_points: int = 2
    num_windows: int = 2
    patch_size: int = 16
    projector_scale: List[Literal["P4",]] = ["P4"]
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_classes: int = 90
    positional_encoding_size: int = 704 // 16
    pretrain_weights: Optional[str] = "rf-detr-large-2026.pth"
    resolution: int = 704
    # Explicit so populate_args and _build_args_from_configs agree.
    # ModelConfig does not define these fields; without them the legacy path
    # picks up populate_args defaults (num_select=100) while the PTL path falls
    # back to TrainConfig.num_select (300), causing a postprocess mismatch.
    num_queries: int = 300
    num_select: int = 300


class RFDETRSegPreviewConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 36
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: Optional[str] = "rf-detr-seg-preview.pt"
    num_classes: int = 90


class RFDETRSegNanoConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 1
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 312
    positional_encoding_size: int = 312 // 12
    num_queries: int = 100
    num_select: int = 100
    pretrain_weights: Optional[str] = "rf-detr-seg-nano.pt"
    num_classes: int = 90


class RFDETRSegSmallConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 4
    patch_size: int = 12
    resolution: int = 384
    positional_encoding_size: int = 384 // 12
    num_queries: int = 100
    num_select: int = 100
    pretrain_weights: Optional[str] = "rf-detr-seg-small.pt"
    num_classes: int = 90


class RFDETRSegMediumConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 5
    patch_size: int = 12
    resolution: int = 432
    positional_encoding_size: int = 432 // 12
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: Optional[str] = "rf-detr-seg-medium.pt"
    num_classes: int = 90


class RFDETRSegLargeConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 5
    patch_size: int = 12
    resolution: int = 504
    positional_encoding_size: int = 504 // 12
    num_queries: int = 200
    num_select: int = 200
    pretrain_weights: Optional[str] = "rf-detr-seg-large.pt"
    num_classes: int = 90


class RFDETRSegXLargeConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 6
    patch_size: int = 12
    resolution: int = 624
    positional_encoding_size: int = 624 // 12
    num_queries: int = 300
    num_select: int = 300
    pretrain_weights: Optional[str] = "rf-detr-seg-xlarge.pt"
    num_classes: int = 90


class RFDETRSeg2XLargeConfig(RFDETRBaseConfig):
    segmentation_head: bool = True
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_windows: int = 2
    dec_layers: int = 6
    patch_size: int = 12
    resolution: int = 768
    positional_encoding_size: int = 768 // 12
    num_queries: int = 300
    num_select: int = 300
    pretrain_weights: Optional[str] = "rf-detr-seg-xxlarge.pt"
    num_classes: int = 90


class TrainConfig(BaseModel):
    """Training hyperparameters and auto-batching configuration.

    Notes:
        * ``auto_batch_target_effective`` is interpreted as the **per-device**
          effective batch size target, i.e. the number of images seen by a
          single process in one optimizer step after accounting for
          ``grad_accum_steps``. In multi-GPU / multi-node runs the global
          effective batch size is therefore:

            ``global_effective_batch = auto_batch_target_effective * devices * num_nodes``

          This avoids silently changing behavior when scaling from single-GPU
          to multi-GPU training.
    """

    lr: float = 1e-4
    lr_encoder: float = 1.5e-4
    batch_size: int | Literal["auto"] = 4
    grad_accum_steps: int = 4
    auto_batch_target_effective: int = 16  # per-device effective batch size target (before devices * num_nodes)
    # Auto-batch probe: worst-case assumptions when batch_size="auto".
    auto_batch_max_targets_per_image: int = 100
    auto_batch_ema_headroom: float = 0.7  # scale safe batch by this when use_ema=True (EMA uses extra memory)
    epochs: int = 100
    resume: Optional[str] = None
    ema_decay: float = 0.993
    ema_tau: int = 100
    lr_drop: int = 100
    checkpoint_interval: int = Field(default=10, ge=1)
    warmup_epochs: float = 0.0
    lr_vit_layer_decay: float = 0.8
    lr_component_decay: float = 0.7
    drop_path: float = 0.0
    group_detr: int = 13
    ia_bce_loss: bool = True
    cls_loss_coef: float = 1.0
    num_select: int = 300
    dataset_file: Literal["coco", "o365", "roboflow", "yolo"] = "roboflow"
    square_resize_div_64: bool = True
    dataset_dir: str
    output_dir: str = "output"
    multi_scale: bool = True
    expanded_scales: bool = True
    do_random_resize_via_padding: bool = False
    use_ema: bool = True
    ema_update_interval: int = 1
    num_workers: int = 2
    weight_decay: float = 1e-4
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_use_ema: bool = False
    progress_bar: Optional[Literal["tqdm", "rich"]] = None  # Progress bar style: "rich", "tqdm", or None to disable.
    tensorboard: bool = True
    wandb: bool = False
    mlflow: bool = False
    clearml: bool = False  # Not yet implemented — reserved for future use.
    project: Optional[str] = None
    run: Optional[str] = None
    class_names: List[str] = None
    run_test: bool = False
    segmentation_head: bool = False
    eval_max_dets: int = 500
    eval_interval: int = 1
    log_per_class_metrics: bool = True
    aug_config: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _warn_deprecated_train_config_fields(self) -> "TrainConfig":
        """Emit DeprecationWarning for fields whose ownership is moving to ModelConfig.

        The following fields are duplicated between ``ModelConfig`` and ``TrainConfig``
        but ``ModelConfig`` is the authoritative source (Item #3, v1.7).  Setting them
        on ``TrainConfig`` is deprecated.  The fields will be removed in v1.9.

        - ``group_detr``: query group count is an architecture decision → ``ModelConfig``
        - ``ia_bce_loss``: loss type is tied to architecture family → ``ModelConfig``
        - ``segmentation_head``: architecture flag → ``ModelConfig``
        - ``num_select``: postprocessor count is an architecture decision → ``ModelConfig``
        """
        _deprecated = ("group_detr", "ia_bce_loss", "segmentation_head", "num_select")
        for field in _deprecated:
            if field in self.model_fields_set:
                # stacklevel=2 points into Pydantic internals; unavoidable with
                # @model_validator(mode="after") in Pydantic v2.
                warnings.warn(
                    f"TrainConfig.{field} is deprecated and will be removed in v1.9. "
                    f"Set {field} on ModelConfig instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        return self

    @field_validator("progress_bar", mode="before")
    @classmethod
    def _coerce_legacy_progress_bar(cls, value: Any) -> Any:
        """Normalize legacy boolean progress_bar values to the new string/None representation.

        This preserves compatibility with older configs where ``progress_bar`` was a bool.
        """
        if isinstance(value, bool):
            return "tqdm" if value else None
        return value

    # Promoted from populate_args() — PTL migration (T4-2).
    # device is intentionally absent: PTL auto-detects accelerator via Trainer(accelerator="auto").
    accelerator: str = "auto"
    clip_max_norm: float = 0.1
    seed: Optional[int] = None
    sync_bn: bool = False
    # strategy maps to PTL Trainer(strategy=...). Common values: "auto", "ddp",
    # "ddp_spawn", "fsdp", "deepspeed". Invalid values surface as PTL errors.
    strategy: str = "auto"
    devices: Union[int, str] = 1
    # num_nodes maps to PTL Trainer(num_nodes=...) for multi-machine training.
    # Single-machine DDP users should leave this at 1 (the default).
    num_nodes: int = 1
    fp16_eval: bool = False
    lr_scheduler: Literal["step", "cosine"] = "step"
    lr_min_factor: float = 0.0
    dont_save_weights: bool = False
    # PTL runtime/perf tuning knobs.
    train_log_sync_dist: bool = False
    train_log_on_step: bool = False
    compute_val_loss: bool = True
    compute_test_loss: bool = True
    pin_memory: Optional[bool] = None
    persistent_workers: Optional[bool] = None
    prefetch_factor: Optional[int] = None

    @field_validator("batch_size", mode="after")
    @classmethod
    def validate_batch_size(cls, v: int | Literal["auto"]) -> int | Literal["auto"]:
        """Validate batch_size is a positive integer or the literal 'auto'."""
        if v == "auto":
            return v
        if v < 1:
            raise ValueError("batch_size must be >= 1, or 'auto'.")
        return v

    @field_validator(
        "grad_accum_steps", "auto_batch_target_effective", "auto_batch_max_targets_per_image", mode="after"
    )
    @classmethod
    def validate_positive_train_steps(cls, v: int) -> int:
        """Validate accumulation, target-effective batch, and max targets are >= 1."""
        if v < 1:
            raise ValueError(
                "grad_accum_steps, auto_batch_target_effective, and auto_batch_max_targets_per_image must be >= 1."
            )
        return v

    @field_validator("auto_batch_ema_headroom", mode="after")
    @classmethod
    def validate_ema_headroom(cls, v: float) -> float:
        """Validate auto_batch_ema_headroom is in (0, 1]."""
        if not (0 < v <= 1.0):
            raise ValueError("auto_batch_ema_headroom must be in (0, 1].")
        return v

    @field_validator("ema_update_interval", "eval_interval", mode="after")
    @classmethod
    def validate_positive_intervals(cls, v: int) -> int:
        """Validate interval fields are >= 1."""
        if v < 1:
            raise ValueError("Interval fields must be >= 1.")
        return v

    @field_validator("prefetch_factor", mode="after")
    @classmethod
    def validate_prefetch_factor(cls, v: Optional[int]) -> Optional[int]:
        """Validate prefetch_factor is None or >= 1."""
        if v is not None and v < 1:
            raise ValueError("prefetch_factor must be >= 1 when provided.")
        return v

    @field_validator("dataset_dir", "output_dir", mode="after")
    @classmethod
    def expand_paths(cls, v: str) -> str:
        """
        Expand user paths (e.g., '~' or paths with separators) but leave simple filenames
        (like 'rf-detr-base.pth') unchanged so they can match hosted model keys.
        """
        if v is None:
            return v
        return os.path.realpath(os.path.expanduser(v))


class SegmentationTrainConfig(TrainConfig):
    num_select: Optional[int] = None
    mask_point_sample_ratio: int = 16
    mask_ce_loss_coef: float = 5.0
    mask_dice_loss_coef: float = 5.0
    cls_loss_coef: float = 5.0
    segmentation_head: bool = True
