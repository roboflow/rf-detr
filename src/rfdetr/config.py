# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------


import os
import warnings
from typing import Any, ClassVar, Dict, List, Literal, Mapping, Optional, TypeAlias, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticUndefined

EncoderName: TypeAlias = Literal["dinov2_windowed_small", "dinov2_windowed_base", "dinov2_registers_windowed_small"]


class PretrainWeightsCompatibilityWarning(UserWarning):
    """Warning emitted when ``ModelConfig`` overrides are likely to prevent the variant's
    published pretrained weights from loading into the model — leaving large portions
    of the model randomly initialized and typically producing much lower accuracy.
    """


def _detect_device() -> str:
    """Detect the best available device **without** initialising the CUDA runtime.

    ``torch.cuda.is_available()`` creates a CUDA driver context that makes
    ``_is_in_bad_fork()`` return ``True`` in child processes.  This breaks
    fork-based DDP strategies (e.g. ``ddp_notebook``) in notebook environments.

    We defer to :func:`torch.accelerator.current_accelerator` (PyTorch ≥ 2.4)
    when available — it queries the driver through NVML without creating a
    primary context.  On older builds we fall back to ``torch.cuda.is_available()``.
    """
    accelerator = getattr(torch, "accelerator", None)
    current_accelerator = getattr(accelerator, "current_accelerator", None)
    if current_accelerator is not None:
        try:
            accel = current_accelerator()
            if accel is not None:
                return str(accel)
            return "cpu"
        except RuntimeError:
            return "cpu"
    # Fallback for PyTorch < 2.4 — this DOES create a CUDA driver context.
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE: str = _detect_device()


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
    encoder: EncoderName
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
    num_channels: int = Field(default=3, ge=1)
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

    @model_validator(mode="after")
    def _sync_pe_with_resolution(self) -> "ModelConfig":
        """Auto-update positional_encoding_size when resolution is explicitly provided.

        When a user provides a custom ``resolution`` at construction time (e.g.,
        ``RFDETRLarge(resolution=640)``), ``positional_encoding_size`` is updated
        proportionally, provided the class-default PE is formula-derived
        (``default_pe == default_resolution // patch_size``).

        Configs with a pretrained-specific PE (e.g., ``RFDETRBaseConfig`` with
        ``positional_encoding_size=37`` for DINOv2's native 518 px grid, while
        ``resolution=560``) are left unchanged.
        """
        if "resolution" not in self.model_fields_set or "positional_encoding_size" in self.model_fields_set:
            return self

        cls = type(self)
        default_resolution = cls.model_fields["resolution"].default
        default_pe = cls.model_fields["positional_encoding_size"].default
        default_patch_size = cls.model_fields["patch_size"].default

        # Skip when any relevant default is not a concrete integer (abstract base
        # class fields have no defaults; required fields use PydanticUndefined,
        # not int).
        if (
            not isinstance(default_resolution, int)
            or not isinstance(default_pe, int)
            or not isinstance(default_patch_size, int)
        ):
            return self

        # Only update PE when the class default is formula-derived from the class
        # default resolution and patch size.
        if default_pe == default_resolution // default_patch_size:
            self.positional_encoding_size = self.resolution // self.patch_size

        return self

    @model_validator(mode="after")
    def _warn_pretrain_compatibility(self) -> "ModelConfig":
        """Warn when overrides are likely to prevent published pretrained weights from loading.

        Three cases:

        1. ``pretrain_weights`` was explicitly set to ``None`` and the variant
           has a non-``None`` default → warn that the model is being initialised
           from scratch.
        2. ``pretrain_weights`` was explicitly set to a non-``None`` custom path
           → suppress the architecture-override check (we cannot know the
           architecture stored in a user-supplied checkpoint at config time).
           The load-time partial-load detector in
           :func:`rfdetr.models.weights.load_pretrain_weights` covers this case
           by inspecting the checkpoint contents directly.
        3. ``pretrain_weights`` is the variant's published default → check
           architecture-affecting fields against the variant defaults and emit
           a single consolidated warning listing every load-breaking override.

        The warning class is :class:`PretrainWeightsCompatibilityWarning` (a
        :class:`UserWarning` subclass), silenceable via the standard
        ``warnings.filterwarnings`` machinery.
        """
        cls = type(self)
        fields_set = self.model_fields_set
        pretrain_user_set = "pretrain_weights" in fields_set

        if pretrain_user_set and self.pretrain_weights is None:
            default_pretrain = cls.model_fields["pretrain_weights"].default
            if default_pretrain is not PydanticUndefined and default_pretrain is not None:
                warnings.warn(
                    f"{cls.__name__} was instantiated with pretrain_weights=None. "
                    f"The model will be initialised from scratch, which typically "
                    f"produces lower accuracy than fine-tuning from the published "
                    f"checkpoint ({default_pretrain!r}).",
                    PretrainWeightsCompatibilityWarning,
                    stacklevel=2,
                )
            return self

        if pretrain_user_set and self.pretrain_weights is not None:
            # Custom checkpoint: architecture overrides may match what the
            # checkpoint was trained with.  Defer to the load-time partial-load
            # detector which can read the file.
            # Exception: when the user explicitly passes the variant's own
            # published-default path string (e.g. ``"rf-detr-nano.pth"``), it
            # IS the published checkpoint — treat it as case 3 so architecture-
            # override checks still apply.  Compare after expand_path so bare
            # filenames resolve to the same cache-dir path as self.pretrain_weights.
            _default_pretrain = cls.model_fields["pretrain_weights"].default
            if _default_pretrain is not None and _default_pretrain is not PydanticUndefined:
                _expanded_default = cls.expand_path(_default_pretrain)
                if self.pretrain_weights != _expanded_default:
                    return self
                # Falls through to case-3 when the user passed the exact variant default.
            else:
                return self

        # `pretrain_weights` is the variant's published default — check
        # architecture overrides against the class defaults.
        # Skip entirely when this variant has no published checkpoint (default
        # is None/PydanticUndefined); warning would reference "(None)" which is
        # misleading and confusing for users of the abstract base config.
        _class_default_pretrain = cls.model_fields["pretrain_weights"].default
        if _class_default_pretrain is None or _class_default_pretrain is PydanticUndefined:
            return self

        overrides: list[tuple[str, Any, Any]] = []

        # Fields that, when explicitly overridden to any value other than the
        # variant default, prevent the published checkpoint from loading cleanly.
        # Includes major architecture knobs, "less obvious" knobs (bbox_reparam,
        # lite_refpoint_refine, layer_norm, two_stage), defense-in-depth for
        # fields that currently raise hard errors (patch_size, segmentation_head),
        # and num_channels (loads via heuristic but result isn't real pretrained
        # weights for the new input domain).
        breaking_fields: tuple[str, ...] = (
            "encoder",
            "hidden_dim",
            "dec_layers",
            "num_windows",
            "sa_nheads",
            "ca_nheads",
            "dec_n_points",
            "out_feature_indexes",
            "projector_scale",
            "bbox_reparam",
            "lite_refpoint_refine",
            "layer_norm",
            "two_stage",
            "patch_size",
            "segmentation_head",
            "num_channels",
        )
        # Fields where only an *increase* above the variant default is load-breaking:
        # num_queries / group_detr add slots whose shape differs — decrease is fine.
        breaking_on_increase: tuple[str, ...] = (
            "num_queries",
            "group_detr",
        )

        for name in breaking_fields:
            if name not in fields_set:
                continue
            field_info = cls.model_fields.get(name)
            if field_info is None or field_info.is_required():
                continue
            default = field_info.default
            if default is PydanticUndefined:
                continue
            current = getattr(self, name)
            if current != default:
                overrides.append((name, current, default))

        for name in breaking_on_increase:
            if name not in fields_set:
                continue
            field_info = cls.model_fields.get(name)
            if field_info is None or field_info.is_required():
                continue
            default = field_info.default
            if default is PydanticUndefined or not isinstance(default, int):
                continue
            current = getattr(self, name)
            if isinstance(current, int) and current > default:
                overrides.append((name, current, default))

        # ``mask_downsample_ratio`` only affects segmentation models — skip on
        # detector-only variants to avoid a misleading "weights won't load" warning.
        if "mask_downsample_ratio" in fields_set and self.segmentation_head:
            _mdr_info = cls.model_fields.get("mask_downsample_ratio")
            if _mdr_info is not None and not _mdr_info.is_required():
                _mdr_default = _mdr_info.default
                if _mdr_default is not PydanticUndefined:
                    _mdr_current = getattr(self, "mask_downsample_ratio")
                    if _mdr_current != _mdr_default:
                        overrides.append(("mask_downsample_ratio", _mdr_current, _mdr_default))

        if overrides:
            default_pretrain = cls.model_fields["pretrain_weights"].default
            lines = "\n".join(
                f"  {name}: {current!r} (variant default: {default!r})" for name, current, default in overrides
            )
            warnings.warn(
                f"{cls.__name__} was instantiated with overrides that differ from the variant "
                f"defaults in ways that prevent the published pretrained weights "
                f"({default_pretrain!r}) from loading correctly:\n"
                f"{lines}\n"
                "Loading the checkpoint with this configuration will leave significant portions "
                "of the model randomly initialised, which typically produces lower accuracy. "
                "To suppress this warning: revert the override(s), pick a variant whose defaults "
                "match, or pass pretrain_weights=None to acknowledge that you intend to train "
                "from scratch.",
                PretrainWeightsCompatibilityWarning,
                stacklevel=2,
            )

        return self

    @field_validator("pretrain_weights", mode="after")
    @classmethod
    def expand_path(cls, v: Optional[str]) -> Optional[str]:
        """Expand and resolve the pretrain_weights path.

        Bare filenames (no directory component, e.g. ``rf-detr-base.pth``) are
        resolved to the model cache directory so weights land in a stable,
        user-configurable location (``~/.roboflow/models`` by default, or the
        path set via the ``RF_HOME`` environment variable) instead of CWD.

        Paths that already contain a directory separator (e.g. ``~/models/x.pth``,
        ``/abs/path/x.pth``, ``models/x.pth``) are normalised with
        ``os.path.realpath`` as before.
        """
        if v is None:
            return v
        expanded = os.path.expanduser(v)
        if not os.path.dirname(expanded):
            # Bare filename → use model cache dir so weights don't land in CWD.
            from rfdetr.assets.model_weights import get_model_cache_dir

            return os.path.join(get_model_cache_dir(), expanded)
        return os.path.realpath(expanded)

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

    encoder: EncoderName = "dinov2_windowed_small"
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
    skip_best_epochs: int = Field(default=0, ge=0)
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
    class_names: Optional[List[str]] = None
    run_test: bool = False
    segmentation_head: bool = False
    eval_max_dets: int = 500
    eval_interval: int = 1
    log_per_class_metrics: bool = True
    aug_config: Optional[Dict[str, Any]] = None
    augmentation_backend: Literal["cpu", "auto", "gpu"] = "cpu"
    save_dataset_grids: bool = False
    notes: Optional[Any] = Field(
        default=None,
        description=(
            "User-defined provenance metadata embedded in best-model .pth checkpoints "
            "under checkpoint['args']['notes'] and in exported ONNX files under the "
            "'rfdetr_notes' metadata property. Accepts any JSON-serialisable value "
            "(string, dict, list, int, float, bool). String values are stored verbatim; "
            "all other types are JSON-encoded."
        ),
    )

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
