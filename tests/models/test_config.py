# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for ``rfdetr.config`` models: validation, defaults, deprecations, and device/precision probes."""

import functools
import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pydantic import Field, ValidationError

import rfdetr.config as config_module
from rfdetr.config import (
    AugmentationBackend,
    KeypointTrainConfig,
    ModelConfig,
    MultiScale,
    PretrainWeightsCompatibilityWarning,
    RFDETRBaseConfig,
    RFDETRLargeConfig,
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
    RFDETRSmallConfig,
    SegmentationTrainConfig,
    TrainConfig,
    _cuda_bf16_supported_on_devices,
    _cuda_supports_native_bf16,
    _cuda_training_device_indices,
    _detect_device,
    _resolve_amp_dtype,
)
from rfdetr.detr import RFDETR
from rfdetr.training import build_trainer


@pytest.fixture
def sample_model_config() -> dict[str, object]:
    """Minimal valid ``ModelConfig`` keyword arguments."""
    return {
        "encoder": "dinov2_windowed_small",
        "out_feature_indexes": [1, 2, 3],
        "dec_layers": 3,
        "projector_scale": ["P3"],
        "hidden_dim": 256,
        "patch_size": 14,
        "num_windows": 2,
        "sa_nheads": 8,
        "ca_nheads": 8,
        "dec_n_points": 4,
        "resolution": 384,
        "positional_encoding_size": 256,
    }


def _make_train_config(tmp_path: Path, **kwargs: object) -> TrainConfig:
    """Build a minimal ``TrainConfig`` rooted at ``tmp_path`` with experiment trackers off.

    Args:
        tmp_path: Directory used for both ``dataset_dir`` and ``output_dir``.
        **kwargs: Field overrides applied on top of the minimal defaults.

    Returns:
        The validated training configuration.

    Examples:
        >>> from pathlib import Path
        >>> _make_train_config(Path("/tmp"), eval_interval=3).eval_interval
        3
    """
    defaults = dict(dataset_dir=str(tmp_path), output_dir=str(tmp_path), tensorboard=False)
    defaults.update(kwargs)
    return TrainConfig(**defaults)


class TestModelConfigValidation:
    """``ModelConfig`` field validation, device normalisation, and path coercion."""

    def test_rejects_unknown_fields(self, sample_model_config) -> None:
        """An unknown constructor field raises a ``ValidationError`` naming it."""
        sample_model_config["unknown"] = "value"

        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(**sample_model_config)

        assert "Unknown parameter(s): 'unknown'" in str(exc_info.value)

    def test_rejects_unknown_attribute_assignment(self, sample_model_config) -> None:
        """Assigning an undeclared attribute on a constructed config raises ``ValueError``."""
        config = ModelConfig(**sample_model_config)

        with pytest.raises(ValueError) as exc_info:
            setattr(config, "unknown", "value")

        assert "Unknown attribute: 'unknown'." in str(exc_info.value)

    def test_accepts_indexed_cuda_device_string(self, sample_model_config) -> None:
        """An indexed CUDA device string such as ``cuda:1`` is stored unchanged."""
        config = ModelConfig(**sample_model_config, device="cuda:1")
        assert config.device == "cuda:1"

    def test_accepts_torch_device(self, sample_model_config) -> None:
        """A ``torch.device`` is normalised to its string form."""
        config = ModelConfig(**sample_model_config, device=torch.device("cuda:2"))
        assert config.device == "cuda:2"

    def test_rejects_non_string_non_torch_device_with_validation_error(self, sample_model_config) -> None:
        """A device that is neither a string nor a ``torch.device`` raises ``ValidationError``."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(**sample_model_config, device=123)

        assert "device must be a string or torch.device." in str(exc_info.value)

    def test_rejects_invalid_device_string(self, sample_model_config) -> None:
        """An unparsable device string raises ``ValidationError`` naming the bad specifier."""
        with pytest.raises(ValidationError) as exc_info:
            ModelConfig(**sample_model_config, device="notadevice")

        assert "Invalid device specifier: 'notadevice'." in str(exc_info.value)

    @pytest.mark.parametrize(
        "encoder",
        [
            "dinov2_windowed_small",
            "dinov2_windowed_base",
            "dinov2_registers_windowed_small",
        ],
    )
    def test_accepts_valid_encoder(self, sample_model_config, encoder: str) -> None:
        """ModelConfig accepts every value in the EncoderName Literal."""
        config = ModelConfig(**{**sample_model_config, "encoder": encoder})
        assert config.encoder == encoder

    def test_rejects_invalid_encoder(self, sample_model_config) -> None:
        """ModelConfig raises ValidationError for encoder strings outside the Literal."""
        with pytest.raises(ValidationError):
            ModelConfig(**{**sample_model_config, "encoder": "dinov2_invalid_backbone"})

    def test_rejects_negative_postprocess_trace_alpha(self, sample_model_config) -> None:
        """ModelConfig rejects negative uncertainty score-fusion exponents."""
        with pytest.raises(ValidationError):
            ModelConfig(**sample_model_config, postprocess_trace_alpha=-0.1)

    def test_postprocess_trace_alpha_defaults_to_keypoint_fusion_value(self, sample_model_config) -> None:
        """ModelConfig defaults to the keypoint uncertainty score-fusion exponent."""
        config = ModelConfig(**sample_model_config)

        assert config.postprocess_trace_alpha == 0.2

    def test_pretrain_weights_absolute_path_realpath_normalised(self, tmp_path) -> None:
        """An absolute pathlib.Path for pretrain_weights is stored as the realpath-normalised string."""
        weights_path = tmp_path / "weights.pth"

        config = RFDETRBaseConfig(pretrain_weights=weights_path)

        assert config.pretrain_weights == os.path.realpath(os.fspath(weights_path))

    @pytest.mark.parametrize(
        "field",
        [
            "dataset_dir",
            "output_dir",
        ],
    )
    def test_train_dir_fields_accept_path(self, tmp_path, field: str) -> None:
        """TrainConfig dataset/output dir fields accept pathlib.Path and store the realpath-normalised string."""
        path = tmp_path / "artifact"
        kwargs = {"dataset_dir": str(tmp_path)}
        kwargs[field] = path

        config = TrainConfig(**kwargs)

        assert getattr(config, field) == os.path.realpath(os.fspath(path))

    def test_accepts_bare_path_object_for_pretrain_weights(self) -> None:
        """Bare pretrained weight Path values resolve the same as bare strings."""
        path_config = RFDETRBaseConfig(pretrain_weights=Path("rf-detr-base.pth"))
        string_config = RFDETRBaseConfig(pretrain_weights="rf-detr-base.pth")

        assert path_config.pretrain_weights == string_config.pretrain_weights

    @pytest.mark.parametrize(
        "value",
        [
            # PTL trainer.fit(ckpt_path=...) sentinels — must pass through verbatim.
            "best",
            "last",
            "hpc",
            "registry:model-name",
            # A relative Path proves os.fspath coercion without realpath resolution.
            pytest.param(Path("checkpoints/last.ckpt"), id="path_object"),
        ],
    )
    def test_resume_coerced_via_fspath_without_realpath(self, value) -> None:
        """``resume`` accepts pathlib.Path and is coerced to ``str`` via ``os.fspath`` only.

        Unlike ``dataset_dir``/``output_dir``, ``resume`` is forwarded verbatim to PyTorch Lightning's
        ``trainer.fit(ckpt_path=...)``, which also accepts sentinels such as ``"last"``. Realpath-normalising it would
        rewrite those sentinels (and relative paths) into spurious absolute paths, so the value must be preserved.
        """
        config = TrainConfig(dataset_dir="/tmp", resume=value)

        assert config.resume == os.fspath(value)


class TestRFDETRBaseConfigEncoder:
    """Encoder field validation on RFDETRBaseConfig (no fixture needed — has defaults)."""

    def test_accepts_registers_windowed_small(self) -> None:
        """RFDETRBaseConfig accepts the new dinov2_registers_windowed_small encoder."""
        config = RFDETRBaseConfig(encoder="dinov2_registers_windowed_small", pretrain_weights=None)
        assert config.encoder == "dinov2_registers_windowed_small"

    def test_rejects_invalid_encoder(self) -> None:
        """RFDETRBaseConfig raises ValidationError for unknown encoder strings."""
        with pytest.raises(ValidationError):
            RFDETRBaseConfig(encoder="not_a_real_encoder", pretrain_weights=None)


class TestModelConfigNumSelect:
    """Unit tests for ModelConfig.num_select per-model default values."""

    @pytest.mark.parametrize(
        "config_class, expected_num_select",
        [
            (RFDETRSegNanoConfig, 100),
            (RFDETRSegSmallConfig, 100),
            (RFDETRSegMediumConfig, 200),
            (RFDETRSegLargeConfig, 200),
            (RFDETRSegXLargeConfig, 300),
            (RFDETRSeg2XLargeConfig, 300),
        ],
    )
    def test_model_config_has_variant_specific_num_select(self, config_class, expected_num_select) -> None:
        """Each segmentation variant config ships its own default ``num_select``."""
        assert config_class().num_select == expected_num_select


class TestTrainConfigRejectsUnknownKwargs:
    """TrainConfig must raise on unknown/typo'd kwargs instead of silently ignoring them (extra='forbid')."""

    def test_typo_kwarg_raises_with_helpful_message(self, tmp_path) -> None:
        """A typo'd kwarg (epoch instead of epochs) raises naming it and listing the available parameters."""
        with pytest.raises(ValidationError) as exc_info:
            TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), epoch=5)

        message = str(exc_info.value)
        assert "Unknown parameter(s): 'epoch'" in message
        assert "Available parameter(s):" in message
        assert "epochs" in message

    @pytest.mark.parametrize(
        "config_class",
        [
            pytest.param(SegmentationTrainConfig, id="segmentation"),
            pytest.param(KeypointTrainConfig, id="keypoint"),
        ],
    )
    def test_subclasses_reject_unknown_kwargs(self, tmp_path, config_class) -> None:
        """TrainConfig subclasses inherit the forbid behaviour."""
        with pytest.raises(ValidationError) as exc_info:
            config_class(dataset_dir=str(tmp_path), output_dir=str(tmp_path), epoch=5)

        assert "Unknown parameter(s): 'epoch'" in str(exc_info.value)

    def test_get_train_config_raises_for_typo_kwarg(self, tmp_path) -> None:
        """The public RFDETR.get_train_config path surfaces the typo instead of swallowing it."""
        stub = SimpleNamespace(_train_config_class=TrainConfig)
        with pytest.raises(ValidationError) as exc_info:
            RFDETR.get_train_config(stub, dataset_dir=str(tmp_path), output_dir=str(tmp_path), epoch=5)

        assert "Unknown parameter(s): 'epoch'" in str(exc_info.value)


class TestTrainConfigMultiScale:
    """`multi_scale` is a `MultiScale` enum; booleans are accepted as aliases and never stored."""

    def test_default_is_batch(self, tmp_path) -> None:
        """The default mode is the per-batch scale draw, matching the old `multi_scale=True` behaviour."""
        assert TrainConfig(dataset_dir=str(tmp_path)).multi_scale is MultiScale.PER_BATCH

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param(True, MultiScale.PER_BATCH, id="legacy-true"),
            pytest.param(False, MultiScale.OFF, id="legacy-false"),
            pytest.param("per-batch", MultiScale.PER_BATCH, id="per-batch"),
            pytest.param("per-sample", MultiScale.PER_SAMPLE, id="per-sample"),
            pytest.param("off", MultiScale.OFF, id="off"),
            pytest.param(MultiScale.PER_SAMPLE, MultiScale.PER_SAMPLE, id="member"),
        ],
    )
    def test_inputs_normalize_to_member(self, tmp_path, value, expected) -> None:
        """Booleans, string values, and members all land on the matching enum member."""
        assert TrainConfig(dataset_dir=str(tmp_path), multi_scale=value).multi_scale is expected

    def test_model_dump_serializes_to_string(self, tmp_path) -> None:
        """`model_dump` yields the plain string value so checkpoint hyperparameters stay JSON-safe."""
        assert (
            TrainConfig(dataset_dir=str(tmp_path), multi_scale="per-sample").model_dump()["multi_scale"] == "per-sample"
        )

    def test_unknown_mode_rejected(self, tmp_path) -> None:
        """A string outside the enum is a validation error."""
        with pytest.raises(ValidationError, match="multi_scale"):
            TrainConfig(dataset_dir=str(tmp_path), multi_scale="image")


class TestTrainConfigT42PromotedFields:
    """T4-2: Promoted fields exist with correct defaults; device field is absent."""

    # --- device field removed ---

    def test_device_not_in_model_fields(self):
        """Device must not appear in TrainConfig.model_fields (PTL auto-detects accelerator)."""
        assert "device" not in TrainConfig.model_fields

    def test_device_kwarg_rejected(self, tmp_path):
        """Passing device= directly to TrainConfig raises (extra='forbid'); RFDETR.train() pops it beforehand."""
        with pytest.raises(ValidationError) as exc_info:
            _make_train_config(tmp_path, device="cpu")

        assert "Unknown parameter(s): 'device'" in str(exc_info.value)

    # --- promoted fields: defaults ---

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("clip_max_norm", 0.1),
            ("seed", None),  # no seeding unless requested
            ("sync_bn", False),
            ("fp16_eval", False),
            ("lr_scheduler", "step"),
            ("lr_scheduler_kwargs", {}),
            ("best_model_metric", "map"),  # backward compatible: rank checkpoints by mAP
            ("optimizer", "adamw"),  # backward compatible
            ("optimizer_kwargs", {}),
            ("dont_save_weights", False),
            ("run_test", False),  # avoids extra full-dataset test passes
            ("eval_interval", 1),  # evaluate each epoch
            ("skip_best_epochs", 0),
            ("ema_update_interval", 1),  # update every step
            ("compute_val_loss", "auto"),  # automatic validation-monitor policy
            ("compute_test_loss", True),
            ("eval_batch_size", None),  # eval loaders inherit the resolved train batch size
            ("pad_targets_to", None),  # keeps the variable-length path CUDA wants
            ("grad_accum_steps", 1),  # opt-in: default effective batch is batch_size alone
            ("eval_base_model", False),  # validation evaluates only the selected model (no extra base forward pass)
            ("eval_masks_head_resolution", False),  # preserves full-resolution mask upsampling
        ],
    )
    def test_promoted_field_default(self, tmp_path, field, expected):
        """Each promoted ``TrainConfig`` field defaults to its documented value and type."""
        value = getattr(_make_train_config(tmp_path), field)
        assert value == expected
        assert type(value) is type(expected)

    def test_clip_max_norm_rejects_negative(self, tmp_path):
        """A negative clip_max_norm fails at construction instead of silently disabling clipping.

        Every consumer gates clipping behind ``> 0`` (``RFDETRModelModule._clip_manual_optimization_gradients``,
        ``FusedAdamWEMA``), so a negative value would train unclipped for the whole run with no error and no warning —
        the failure this boundary constraint turns into a construction error.
        """
        with pytest.raises(ValidationError, match="clip_max_norm"):
            _make_train_config(tmp_path, clip_max_norm=-0.1)

    def test_clip_max_norm_accepts_zero(self, tmp_path):
        """clip_max_norm=0.0 stays legal as the documented way to disable gradient clipping.

        The constraint has to be ``ge``, not ``gt``: callers already pass 0.0 to opt out of clipping (e.g. the keypoint
        DDP training test), so rejecting it would break a supported configuration.
        """
        assert _make_train_config(tmp_path, clip_max_norm=0.0).clip_max_norm == pytest.approx(0.0)

    def test_best_model_metric_accepts_mar(self, tmp_path):
        """best_model_metric accepts 'mar' to rank checkpoints/early-stop by recall instead of mAP."""
        assert _make_train_config(tmp_path, best_model_metric="mar").best_model_metric == "mar"

    def test_best_model_metric_rejects_invalid_value(self, tmp_path):
        """best_model_metric rejects any value outside the 'map'/'mar' literal."""
        with pytest.raises(ValidationError):
            _make_train_config(tmp_path, best_model_metric="f1")

    @pytest.mark.parametrize(
        "value",
        [
            True,
            False,
            "auto",
        ],
    )
    def test_compute_val_loss_accepts_bool_or_auto(self, tmp_path, value):
        """compute_val_loss preserves explicit boolean and automatic policies."""
        assert _make_train_config(tmp_path, compute_val_loss=value).compute_val_loss == value

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("true", True),
            ("false", False),
            ("yes", True),
            ("no", False),
            ("auto", "auto"),
        ],
    )
    def test_compute_val_loss_coerces_string_aliases(self, tmp_path, raw, expected):
        """compute_val_loss silently coerces common string aliases via pydantic's lax bool parsing.

        The bool | Literal["auto"] union tries the bool arm first, so "yes"/"no"/"true"/"false" coerce silently instead
        of raising; "auto" instead matches the Literal arm unchanged. Pinning this behavior catches a pydantic upgrade
        that changes union member resolution order.
        """
        assert _make_train_config(tmp_path, compute_val_loss=raw).compute_val_loss == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "Auto",
            "AUTO",
        ],
    )
    def test_compute_val_loss_rejects_case_variant_of_auto(self, tmp_path, raw):
        """compute_val_loss rejects any casing of 'auto' other than the exact lowercase literal.

        Literal["auto"] is case-strict and these variants don't match the bool arm's alias set either, so pydantic must
        raise rather than silently normalizing casing.
        """
        with pytest.raises(ValidationError):
            _make_train_config(tmp_path, compute_val_loss=raw)

    # --- promoted fields: accept explicit values ---

    @pytest.mark.parametrize(
        "field, value",
        [
            ("clip_max_norm", 0.5),
            ("seed", 42),
            ("sync_bn", True),
            ("fp16_eval", True),
            pytest.param("lr_scheduler_kwargs", {"min_factor": 0.01}, id="lr_scheduler_kwargs"),
            ("lr_scheduler_interval", "epoch"),
            ("lr_scheduler_monitor", "train/loss"),
            ("optimizer", "sgd"),
            pytest.param("optimizer_kwargs", {"betas": (0.9, 0.99)}, id="optimizer_kwargs"),
            ("dont_save_weights", True),
            ("run_test", True),
            ("eval_interval", 3),
            ("skip_best_epochs", 3),
            ("ema_update_interval", 4),
            ("eval_base_model", True),
            ("eval_masks_head_resolution", True),
            ("compute_test_loss", False),
            ("train_log_sync_dist", True),
            ("train_log_on_step", True),
            ("log_per_class_metrics", False),
            ("prefetch_factor", 4),
            ("pin_memory", False),
            ("persistent_workers", False),
        ],
    )
    def test_promoted_field_accepts_explicit_value(self, tmp_path, field, value):
        """Each promoted field accepts an explicit value."""
        tc = _make_train_config(tmp_path, **{field: value})
        assert getattr(tc, field) == value

    def test_lr_scheduler_rejects_invalid_value(self, tmp_path):
        """lr_scheduler must reject values other than 'step' and 'cosine'."""
        with pytest.raises((ValueError, ValidationError)):
            _make_train_config(tmp_path, lr_scheduler="cyclic")

    def test_optimizer_rejects_empty_name(self, tmp_path):
        """Optimizer must be a non-empty name."""
        with pytest.raises((ValueError, ValidationError)):
            _make_train_config(tmp_path, optimizer="  ")

    @pytest.mark.parametrize(
        "optimizer",
        [
            "lion",
            # Pytorch-optimizer names are not selectable by short name (use an import path).
            "pytorch_optimizer:lion",
        ],
    )
    def test_optimizer_rejects_non_native_short_name(self, tmp_path, optimizer):
        """A bare short name that does not resolve to a torch.optim optimizer is rejected at config time."""
        with pytest.raises((ValueError, ValidationError), match="native optimizer"):
            _make_train_config(tmp_path, optimizer=optimizer)

    @pytest.mark.parametrize(
        "reserved_key",
        [
            "params",
            "lr",
            "weight_decay",
            "fused",
        ],
    )
    def test_optimizer_kwargs_reject_reserved_keys(self, tmp_path, reserved_key):
        """optimizer_kwargs must not override RF-DETR-managed optimizer arguments."""
        with pytest.raises((ValueError, ValidationError)):
            _make_train_config(tmp_path, optimizer_kwargs={reserved_key: 1})

    def test_optimizer_accepts_dotted_import_path(self, tmp_path):
        """A dotted import path is accepted verbatim as an explicit optimizer."""
        assert _make_train_config(tmp_path, optimizer="torch.optim.AdamW").optimizer == "torch.optim.AdamW"

    def test_dotted_optimizer_kwargs_allow_managed_keys(self, tmp_path):
        """Explicit (dotted) optimizers may pass otherwise-managed keys such as weight_decay."""
        tc = _make_train_config(tmp_path, optimizer="torch.optim.AdamW", optimizer_kwargs={"weight_decay": 0.1})
        assert tc.optimizer_kwargs == {"weight_decay": 0.1}

    def test_callable_class_desugars_to_dotted_path(self, tmp_path):
        """A plain optimizer class desugars to its canonical dotted import path for serialization."""
        expected_path = f"{torch.optim.AdamW.__module__}.{torch.optim.AdamW.__qualname__}"
        tc = _make_train_config(tmp_path, optimizer=torch.optim.AdamW)
        assert tc.optimizer == expected_path
        assert tc.optimizer_kwargs == {}

    def test_partial_optimizer_desugars_to_path_and_kwargs(self, tmp_path):
        """A functools.partial desugars to a dotted path plus its keyword arguments."""
        expected_path = f"{torch.optim.AdamW.__module__}.{torch.optim.AdamW.__qualname__}"
        tc = _make_train_config(tmp_path, optimizer=functools.partial(torch.optim.AdamW, weight_decay=0.1))
        assert tc.optimizer == expected_path
        assert tc.optimizer_kwargs == {"weight_decay": 0.1}

    def test_callable_optimizer_kwargs_are_ignored_with_warning(self, tmp_path):
        """optimizer_kwargs are ignored (with a warning) when optimizer is a callable."""
        with pytest.warns(UserWarning, match="optimizer_kwargs is ignored"):
            tc = _make_train_config(
                tmp_path,
                optimizer=functools.partial(torch.optim.AdamW, weight_decay=0.1),
                optimizer_kwargs={"betas": (0.9, 0.99)},
            )
        assert tc.optimizer_kwargs == {"weight_decay": 0.1}

    def test_non_reconstructable_optimizer_warns(self, tmp_path):
        """A lambda optimizer cannot be serialized and warns while staying an in-memory callable."""
        with pytest.warns(UserWarning, match="cannot be saved"):
            tc = _make_train_config(tmp_path, optimizer=lambda params: torch.optim.AdamW(params))
        assert callable(tc.optimizer) and not isinstance(tc.optimizer, str)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("eval_interval", 0),
            ("skip_best_epochs", -1),
            ("ema_update_interval", 0),
            ("prefetch_factor", 0),
        ],
    )
    def test_interval_and_prefetch_reject_non_positive_values(self, tmp_path, field, value):
        """Eval/EMA intervals and prefetch_factor must be >= 1 when provided."""
        with pytest.raises((ValueError, ValidationError)):
            _make_train_config(tmp_path, **{field: value})

    def test_batch_size_auto_is_accepted(self, tmp_path):
        """batch_size accepts the special 'auto' value."""
        tc = _make_train_config(tmp_path, batch_size="auto")
        assert tc.batch_size == "auto"

    def test_fp8_rejects_auto_batch(self, tmp_path: Path) -> None:
        """The ordinary autocast probe cannot size a Transformer Engine model."""
        with pytest.raises(ValueError, match="FP8.*explicit.*batch_size"):
            _make_train_config(tmp_path, batch_size="auto", amp_dtype="fp8")

    @pytest.mark.parametrize("amp_dtype", ["auto", "bf16", "fp16"])
    def test_other_precisions_allow_auto_batch(self, tmp_path: Path, amp_dtype: str) -> None:
        """Existing autocast modes retain automatic batch sizing."""
        assert _make_train_config(tmp_path, batch_size="auto", amp_dtype=amp_dtype).batch_size == "auto"

    def test_fp8_allows_explicit_batch(self, tmp_path: Path) -> None:
        """An explicit FP8 micro-batch does not need the unsupported probe."""
        assert _make_train_config(tmp_path, batch_size=1, amp_dtype="fp8").batch_size == 1

    @pytest.mark.parametrize(
        "field,value",
        [
            ("batch_size", 0),
            ("grad_accum_steps", 0),
            ("auto_batch_target_effective", 0),
            ("auto_batch_max_targets_per_image", 0),
        ],
    )
    def test_auto_batch_related_fields_reject_non_positive_values(self, tmp_path, field, value):
        """batch/accum/target-effective/max_targets fields must be >= 1 (except batch_size='auto')."""
        with pytest.raises((ValueError, ValidationError)):
            _make_train_config(tmp_path, **{field: value})

    @pytest.mark.parametrize(
        "eval_batch_size",
        [
            0,
            -1,
        ],
    )
    def test_eval_batch_size_rejects_non_positive_values(self, tmp_path: Path, eval_batch_size: int) -> None:
        """eval_batch_size must be >= 1 when provided."""
        with pytest.raises(ValidationError, match=r"eval_batch_size\s+Value error, eval_batch_size must be >= 1"):
            _make_train_config(tmp_path, eval_batch_size=eval_batch_size)

    def test_eval_batch_size_rejects_auto_sentinel(self, tmp_path: Path) -> None:
        """eval_batch_size rejects batch_size's automatic-sizing sentinel."""
        with pytest.raises(ValidationError, match=r"eval_batch_size\s+Input should be a valid integer"):
            _make_train_config(tmp_path, eval_batch_size="auto")

    @pytest.mark.parametrize(
        "pad_targets_to",
        [
            0,
            -1,
        ],
    )
    def test_pad_targets_to_rejects_non_positive_values(self, tmp_path: Path, pad_targets_to: int) -> None:
        """pad_targets_to must be >= 1 when provided, caught at construction, not at first collate."""
        with pytest.raises(
            ValidationError, match=r"pad_targets_to\s+Value error, pad_targets_to must be a positive integer"
        ):
            _make_train_config(tmp_path, pad_targets_to=pad_targets_to)

    @pytest.mark.parametrize("ema_headroom", [0.0, 1.5])
    def test_auto_batch_ema_headroom_must_be_in_open_one(self, tmp_path, ema_headroom):
        """auto_batch_ema_headroom must be in (0, 1]."""
        with pytest.raises((ValueError, ValidationError)):
            _make_train_config(tmp_path, auto_batch_ema_headroom=ema_headroom)

    def test_eval_ema_only_requires_use_ema(self, tmp_path):
        """eval_ema_only=True with use_ema=False must raise — no EMA model exists to evaluate.

        The flag is deprecated but still validated: a contradictory pair must fail loudly rather than
        be silently absorbed by the deprecation shim.
        """
        with pytest.raises(ValidationError, match="eval_ema_only"):
            with pytest.warns(FutureWarning):
                _make_train_config(tmp_path, eval_ema_only=True, use_ema=False)

    def test_eval_ema_only_accepted_with_use_ema(self, tmp_path):
        """eval_ema_only=True is accepted when use_ema=True (the default), and warns that it is deprecated.

        Evaluating only the selected model is now the default, so the flag is inert. Existing scripts that set it must
        keep working through the 0.3-cycle deprecation window rather than failing on an unknown field.
        """
        with pytest.warns(FutureWarning, match="eval_ema_only"):
            tc = _make_train_config(tmp_path, eval_ema_only=True, use_ema=True)
        assert tc.eval_ema_only is True

    def test_legacy_eval_ema_only_false_maps_to_base_model(self, tmp_path):
        """An old explicit ``eval_ema_only=False`` input retains its prior base-plus-EMA behavior."""
        with pytest.warns(FutureWarning, match="eval_ema_only"):
            tc = _make_train_config(tmp_path, eval_ema_only=False, use_ema=True)

        assert tc.eval_base_model is True

    def test_dumped_eval_ema_only_config_reloads_without_warning(self, tmp_path):
        """A new config dump carries the migrated policy and can reload without repeating the deprecation warning."""
        with pytest.warns(FutureWarning, match="eval_ema_only"):
            original = _make_train_config(tmp_path, eval_ema_only=True, use_ema=True)

        dumped = original.model_dump()
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            reloaded = TrainConfig(**dumped)

        assert reloaded.eval_base_model is False

    def test_eval_ema_only_defaults_to_false_without_warning(self, tmp_path):
        """eval_ema_only defaults to False and a config that never sets it must not warn.

        Round-tripping a dumped config carries every field explicitly at its default; warning on the default value would
        make every reload noisy.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            tc = _make_train_config(tmp_path)
        assert tc.eval_ema_only is False

    def test_eval_base_model_is_accepted_without_ema(self, tmp_path):
        """eval_base_model=True with use_ema=False must not raise — the base model is evaluated either way.

        A default must be inert in every legal configuration; the opt-in that reverses it has to be too, so no cross-
        field validator may reject this pair.
        """
        tc = _make_train_config(tmp_path, eval_base_model=True, use_ema=False)
        assert tc.eval_base_model is True

    def test_eval_ema_only_conflicts_with_eval_base_model(self, tmp_path):
        """eval_ema_only=True together with eval_base_model=True must raise — the two requests contradict.

        The deprecated flag asks for EMA-only evaluation while the new opt-in asks for the base model as well; silently
        picking a winner would give one of the two settings no effect.
        """
        with pytest.raises(ValidationError, match="eval_base_model"):
            with pytest.warns(FutureWarning):
                _make_train_config(tmp_path, eval_ema_only=True, eval_base_model=True)


class TestResolveAmpDtype:
    """``amp_dtype`` is the live AMP authority; ``ModelConfig.amp`` is a deprecated fallback."""

    def test_default_resolves_to_auto(self, tmp_path):
        """Neither field touched: AMP stays on with the auto-selected dtype."""
        assert _resolve_amp_dtype(RFDETRNanoConfig(), _make_train_config(tmp_path)) == "auto"

    def test_explicit_none_disables_amp(self, tmp_path):
        """amp_dtype=None is the supported way to train in full fp32."""
        assert _resolve_amp_dtype(RFDETRNanoConfig(), _make_train_config(tmp_path, amp_dtype=None)) is None

    def test_explicit_amp_dtype_outranks_deprecated_amp_false(self, tmp_path):
        """A caller-set amp_dtype is never silently disabled by a stale amp=False."""
        assert _resolve_amp_dtype(RFDETRNanoConfig(amp=False), _make_train_config(tmp_path, amp_dtype="bf16")) == "bf16"

    def test_explicit_fp8_outranks_deprecated_amp_false(self, tmp_path):
        """Fp8 is no exception to the precedence rule; hardware checks, not amp, gate it downstream."""
        resolved = _resolve_amp_dtype(
            RFDETRNanoConfig(amp=False), _make_train_config(tmp_path, amp_dtype="fp8", batch_size=4)
        )
        assert resolved == "fp8"

    def test_deprecated_amp_false_applies_when_amp_dtype_is_default(self, tmp_path):
        """Legacy amp=False keeps working while amp_dtype is untouched."""
        with pytest.warns(FutureWarning, match="ModelConfig.amp is deprecated"):
            assert _resolve_amp_dtype(RFDETRNanoConfig(amp=False), _make_train_config(tmp_path)) is None

    def test_deprecated_amp_false_survives_a_train_config_round_trip(self, tmp_path):
        """A reloaded config carries amp_dtype explicitly; that must not silently void the legacy toggle."""
        reloaded = TrainConfig(**_make_train_config(tmp_path).model_dump())
        with pytest.warns(FutureWarning, match="ModelConfig.amp is deprecated"):
            assert _resolve_amp_dtype(RFDETRNanoConfig(amp=False), reloaded) is None

    def test_amp_true_does_not_warn(self, tmp_path):
        """The default amp=True is not a deprecated usage and must stay silent."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert _resolve_amp_dtype(RFDETRNanoConfig(), _make_train_config(tmp_path)) == "auto"

    def test_none_is_accepted_by_validation(self, tmp_path):
        """None is a valid amp_dtype, not an unknown value coerced back to 'auto'."""
        assert _make_train_config(tmp_path, amp_dtype=None).amp_dtype is None


class TestDeprecatedFp16Eval:
    """``fp16_eval`` is inert and superseded by ``amp_dtype``."""

    def test_explicit_true_warns(self, tmp_path):
        """Setting the flag surfaces that it does nothing rather than silently ignoring it."""
        with pytest.warns(FutureWarning, match="fp16_eval is deprecated"):
            _make_train_config(tmp_path, fp16_eval=True)

    def test_default_does_not_warn(self, tmp_path):
        """The untouched default must stay silent, including on a dumped-config reload."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            TrainConfig(**_make_train_config(tmp_path).model_dump())


class TestTrainConfigLRScheduler:
    """Configurable LR-scheduler surface: preset validation, explicit paths/callables, and deprecation folding."""

    @pytest.mark.parametrize("preset", ["step", "cosine"])
    def test_lr_scheduler_accepts_managed_preset(self, tmp_path, preset):
        """Both managed presets are accepted as bare names."""
        assert _make_train_config(tmp_path, lr_scheduler=preset).lr_scheduler == preset

    def test_lr_scheduler_rejects_empty_name(self, tmp_path):
        """lr_scheduler must be a non-empty name."""
        with pytest.raises((ValueError, ValidationError)):
            _make_train_config(tmp_path, lr_scheduler="  ")

    def test_lr_scheduler_rejects_unknown_bare_name(self, tmp_path):
        """A bare name that is not a managed preset is rejected (steer to a dotted path)."""
        with pytest.raises((ValueError, ValidationError), match="managed preset"):
            _make_train_config(tmp_path, lr_scheduler="StepLR")

    def test_lr_scheduler_accepts_dotted_import_path(self, tmp_path):
        """A dotted import path is accepted verbatim as an explicit scheduler."""
        tc = _make_train_config(
            tmp_path, lr_scheduler="torch.optim.lr_scheduler.StepLR", lr_scheduler_kwargs={"step_size": 5}
        )
        assert tc.lr_scheduler == "torch.optim.lr_scheduler.StepLR"

    def test_compute_val_loss_false_rejects_plateau_loss_monitor(self, tmp_path):
        """A plateau scheduler monitoring val/loss cannot disable the metric it requires."""
        with pytest.raises(ValidationError, match="compute_val_loss=False requires a non-val/loss monitor"):
            _make_train_config(
                tmp_path,
                compute_val_loss=False,
                lr_scheduler="torch.optim.lr_scheduler.ReduceLROnPlateau",
                lr_scheduler_monitor="val/loss",
            )

    def test_callable_class_desugars_to_dotted_path(self, tmp_path):
        """A plain scheduler class desugars to its canonical dotted import path for serialization."""
        scheduler_class = torch.optim.lr_scheduler.StepLR
        expected_path = f"{scheduler_class.__module__}.{scheduler_class.__qualname__}"
        tc = _make_train_config(tmp_path, lr_scheduler=scheduler_class)
        assert tc.lr_scheduler == expected_path
        assert tc.lr_scheduler_kwargs == {}

    def test_partial_scheduler_desugars_to_path_and_kwargs(self, tmp_path):
        """A functools.partial desugars to a dotted path plus its keyword arguments."""
        scheduler_class = torch.optim.lr_scheduler.StepLR
        expected_path = f"{scheduler_class.__module__}.{scheduler_class.__qualname__}"
        tc = _make_train_config(tmp_path, lr_scheduler=functools.partial(scheduler_class, step_size=7))
        assert tc.lr_scheduler == expected_path
        assert tc.lr_scheduler_kwargs == {"step_size": 7}

    def test_callable_scheduler_kwargs_are_ignored_with_warning(self, tmp_path):
        """lr_scheduler_kwargs are ignored (with a warning) when lr_scheduler is a callable."""
        with pytest.warns(UserWarning, match="lr_scheduler_kwargs is ignored"):
            tc = _make_train_config(
                tmp_path,
                lr_scheduler=functools.partial(torch.optim.lr_scheduler.StepLR, step_size=7),
                lr_scheduler_kwargs={"gamma": 0.5},
            )
        assert tc.lr_scheduler_kwargs == {"step_size": 7}

    def test_non_reconstructable_scheduler_warns(self, tmp_path):
        """A lambda scheduler cannot be serialized and warns while staying an in-memory callable."""
        with pytest.warns(UserWarning, match="cannot be saved"):
            tc = _make_train_config(
                tmp_path, lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, 5)
            )
        assert callable(tc.lr_scheduler) and not isinstance(tc.lr_scheduler, str)

    @pytest.mark.parametrize("field", ["lr_drop", "lr_min_factor"])
    def test_removed_lr_fields_are_rejected(self, tmp_path, field):
        """The v1.9-deprecated top-level LR fields were removed in v1.11 and are now unknown kwargs."""
        with pytest.raises(ValidationError, match=field):
            _make_train_config(tmp_path, **{field: 1})

    def test_managed_preset_rejects_unknown_kwargs(self, tmp_path):
        """Managed presets reject lr_scheduler_kwargs keys they do not consume (mirrors optimizer_kwargs)."""
        with pytest.raises((ValueError, ValidationError), match="unknown key"):
            _make_train_config(tmp_path, lr_scheduler="cosine", lr_scheduler_kwargs={"minfactor": 0.1})

    def test_managed_preset_accepts_consumed_kwargs(self, tmp_path):
        """Managed presets accept the kwargs they consume (min_factor, lr_drop)."""
        tc = _make_train_config(tmp_path, lr_scheduler="cosine", lr_scheduler_kwargs={"min_factor": 0.1, "lr_drop": 50})
        assert tc.lr_scheduler_kwargs == {"min_factor": pytest.approx(0.1), "lr_drop": 50}

    def test_explicit_scheduler_round_trips_through_model_dump(self, tmp_path):
        """An explicit dotted scheduler and its kwargs survive a model_dump() -> reload round-trip unchanged."""
        original = _make_train_config(
            tmp_path,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",
            lr_scheduler_kwargs={"step_size": 30, "gamma": 0.1},
        )
        reloaded = TrainConfig(**original.model_dump())
        assert reloaded.lr_scheduler == "torch.optim.lr_scheduler.StepLR"
        assert reloaded.lr_scheduler_kwargs == {"step_size": 30, "gamma": pytest.approx(0.1)}

    def test_partial_with_positional_args_is_not_serializable(self, tmp_path):
        """A functools.partial with positional args cannot be desugared; it warns and stays an in-memory callable."""
        with pytest.warns(UserWarning, match="cannot be saved"):
            tc = _make_train_config(tmp_path, lr_scheduler=functools.partial(torch.optim.lr_scheduler.StepLR, 5))
        assert callable(tc.lr_scheduler) and not isinstance(tc.lr_scheduler, str)

    def test_partial_with_non_json_kwargs_is_not_serializable(self, tmp_path):
        """A functools.partial carrying non-JSON kwargs cannot be desugared; it warns and stays a callable."""
        with pytest.warns(UserWarning, match="cannot be saved"):
            tc = _make_train_config(
                tmp_path, lr_scheduler=functools.partial(torch.optim.lr_scheduler.StepLR, gamma=object())
            )
        assert callable(tc.lr_scheduler) and not isinstance(tc.lr_scheduler, str)


class TestBuildTrainerUsesRealFields:
    """build_trainer() must read clip_max_norm, seed, sync_bn from real TrainConfig fields."""

    def _tc(self, tmp_path, **kwargs):
        defaults = dict(
            dataset_dir=str(tmp_path),
            output_dir=str(tmp_path),
            tensorboard=False,
            wandb=False,
            mlflow=False,
            clearml=False,
            use_ema=False,
        )
        defaults.update(kwargs)
        return TrainConfig(**defaults)

    def _kp_tc(self, tmp_path, **kwargs):
        defaults = dict(
            dataset_dir=str(tmp_path),
            output_dir=str(tmp_path),
            tensorboard=False,
            wandb=False,
            mlflow=False,
            clearml=False,
            use_ema=False,
        )
        defaults.update(kwargs)
        return KeypointTrainConfig(**defaults)

    def _mc(self, **kwargs):
        defaults = dict(pretrain_weights=None, device="cpu", num_classes=3)
        defaults.update(kwargs)
        return RFDETRBaseConfig(**defaults)

    def test_clip_max_norm_forwarded_to_trainer_for_detection(self, tmp_path):
        """Detection models use Lightning's automatic optimization, so ``gradient_clip_val`` flows through to the
        Trainer from ``TrainConfig.clip_max_norm`` unchanged."""
        trainer = build_trainer(self._tc(tmp_path, clip_max_norm=0.25), self._mc())
        assert trainer.gradient_clip_val == pytest.approx(0.25)

    def test_clip_max_norm_owned_by_model_module_for_keypoints(self, tmp_path):
        """Keypoint models use manual optimization; trainer-owned clipping is disabled and ``clip_max_norm`` is applied
        inside ``RFDETRModelModule.on_before_optimizer_step`` instead."""
        trainer = build_trainer(
            self._kp_tc(tmp_path, clip_max_norm=0.25),
            self._mc(use_grouppose_keypoints=True),
        )
        assert trainer.gradient_clip_val is None

    def test_seed_not_applied_in_build_trainer_factory(self, mocker, tmp_path):
        """Seeding is deferred to RFDETRModule.on_fit_start, not build_trainer()."""
        mock_seed = mocker.patch("pytorch_lightning.seed_everything")

        build_trainer(self._tc(tmp_path, seed=99), self._mc())

        mock_seed.assert_not_called()

    def test_sync_bn_forwarded_to_trainer(self, mocker, tmp_path):
        """sync_batchnorm=True is passed to Trainer when TrainConfig.sync_bn is True."""
        captured_kwargs = {}

        real_trainer_init = __import__("pytorch_lightning").Trainer.__init__

        def _capture_init(self_t, **kwargs):
            captured_kwargs.update(kwargs)
            real_trainer_init(self_t, **kwargs)

        mocker.patch("rfdetr.training.trainer.Trainer.__init__", _capture_init)

        build_trainer(self._tc(tmp_path, sync_bn=True), self._mc())

        assert captured_kwargs.get("sync_batchnorm") is True


class TestSyncPEWithResolutionAtConstruction:
    """Tests for the _sync_pe_with_resolution model_validator.

    When a user provides a custom resolution at construction time (e.g., ``RFDETRLarge(resolution=640)``),
    positional_encoding_size must be updated proportionally for configs where the default PE is formula-derived
    (``default_pe == default_resolution // patch_size``).
    """

    @pytest.mark.parametrize(
        "config_cls, new_resolution, expected_pe",
        [
            pytest.param(RFDETRLargeConfig, 640, 640 // 16, id="large_640"),
            pytest.param(RFDETRLargeConfig, 576, 576 // 16, id="large_576"),
            pytest.param(RFDETRSmallConfig, 640, 640 // 16, id="small_640"),
            pytest.param(RFDETRMediumConfig, 640, 640 // 16, id="medium_640"),
            pytest.param(RFDETRNanoConfig, 416, 416 // 16, id="nano_416"),
            pytest.param(RFDETRSegNanoConfig, 360, 360 // 12, id="seg_nano_360"),
            pytest.param(RFDETRSegSmallConfig, 480, 480 // 12, id="seg_small_480"),
            pytest.param(RFDETRSegMediumConfig, 480, 480 // 12, id="seg_medium_480"),
            pytest.param(RFDETRSegLargeConfig, 576, 576 // 12, id="seg_large_576"),
            pytest.param(RFDETRSegXLargeConfig, 576, 576 // 12, id="seg_xlarge_576"),
            pytest.param(RFDETRSeg2XLargeConfig, 720, 720 // 12, id="seg_2xlarge_720"),
        ],
    )
    def test_positional_encoding_size_updated_for_formula_derived_configs(
        self,
        config_cls: type,
        new_resolution: int,
        expected_pe: int,
    ) -> None:
        """PE is auto-derived from the custom resolution for formula-derived model configs."""
        cfg = config_cls(resolution=new_resolution, pretrain_weights=None)
        assert cfg.positional_encoding_size == expected_pe

    def test_explicit_positional_encoding_size_is_not_overridden(self) -> None:
        """When positional_encoding_size is explicitly provided, the validator must not override it."""
        cfg = RFDETRLargeConfig(resolution=640, positional_encoding_size=50, pretrain_weights=None)
        assert cfg.positional_encoding_size == 50

    def test_default_resolution_preserves_default_pe(self) -> None:
        """Constructing with default resolution (no explicit resolution) must not change PE."""
        cfg = RFDETRLargeConfig(pretrain_weights=None)
        assert cfg.resolution == 704
        assert cfg.positional_encoding_size == 44  # 704 // 16


class TestDetectDevice:
    """Tests for _detect_device() covering PyTorch accelerator detection paths."""

    @pytest.fixture
    def mock_torch(self, mocker):
        """Replace the ``torch`` module seen by ``rfdetr.config`` with a mock."""
        return mocker.patch("rfdetr.config.torch")

    def test_falls_back_to_cuda_when_accelerator_module_absent(self, mocker, mock_torch) -> None:
        """Returns 'cuda' via legacy fallback when torch.accelerator lacks current_accelerator (PyTorch < 2.4)."""
        mock_torch.accelerator = mocker.MagicMock(spec=[])  # no current_accelerator → hasattr returns False → fallback
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False
        assert _detect_device() == "cuda"

    def test_returns_cpu_when_current_accelerator_raises(self, mock_torch) -> None:
        """Returns 'cpu' directly from the except handler when current_accelerator() raises RuntimeError."""
        mock_torch.accelerator.current_accelerator.side_effect = RuntimeError("no device")
        assert _detect_device() == "cpu"

    def test_returns_cpu_when_no_gpu_available(self, mocker, mock_torch) -> None:
        """Returns 'cpu' when accelerator is absent and neither CUDA nor MPS is available."""
        mock_torch.accelerator = mocker.MagicMock(spec=[])  # no current_accelerator → fallback branch
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        assert _detect_device() == "cpu"

    def test_returns_cpu_when_accelerator_compiled_in_but_unavailable(self, mock_torch) -> None:
        """Returns 'cpu' when torch was compiled with CUDA but no driver is present at runtime.

        Without ``check_available=True``, ``current_accelerator()`` reports the compile-time accelerator, so the default
        CUDA wheel on a driverless machine yields ``device("cuda")`` and every model build crashes with "Found no NVIDIA
        driver". The runtime availability check must win.
        """

        def fake_current_accelerator(check_available: bool = False) -> "torch.device | None":
            return None if check_available else torch.device("cuda")

        mock_torch.accelerator.current_accelerator = fake_current_accelerator
        assert _detect_device() == "cpu"

    def test_returns_accelerator_when_runtime_available(self, mock_torch) -> None:
        """Returns the accelerator when it passes the runtime availability check."""

        def fake_current_accelerator(check_available: bool = False) -> "torch.device | None":
            return torch.device("cuda") if check_available else None

        mock_torch.accelerator.current_accelerator = fake_current_accelerator
        assert _detect_device() == "cuda"

    @pytest.mark.parametrize(("accelerator_available", "expected"), [(False, "cpu"), (True, "cuda")])
    def test_legacy_signature_follows_is_available(
        self, mock_torch, accelerator_available: bool, expected: str
    ) -> None:
        """Without ``check_available`` (PyTorch < 2.7) the accelerator is kept only if ``is_available()`` confirms
        it."""

        def legacy_current_accelerator() -> "torch.device":
            return torch.device("cuda")

        mock_torch.accelerator.current_accelerator = legacy_current_accelerator
        mock_torch.accelerator.is_available.return_value = accelerator_available
        assert _detect_device() == expected

    def test_legacy_signature_runtime_error_on_fallback_returns_cpu(self, mock_torch) -> None:
        """Outer RuntimeError handler catches error from legacy fallback call.

        Control-flow: ``current_accelerator(check_available=True)`` raises ``TypeError`` (inner except),
        then ``current_accelerator()`` raises ``RuntimeError`` (outer except catches) → ``"cpu"``.
        """
        call_count = 0

        def raises_on_fallback(**kwargs: object) -> "torch.device":
            nonlocal call_count
            call_count += 1
            if "check_available" in kwargs:
                raise TypeError("unexpected keyword argument 'check_available'")
            raise RuntimeError("NVML error on legacy fallback")

        mock_torch.accelerator.current_accelerator = raises_on_fallback
        assert _detect_device() == "cpu"


class TestCudaTrainingDeviceIndicesClamping:
    """``_cuda_training_device_indices`` returns only indices this host can actually see.

    Every index it returns is handed straight to a per-device probe such as ``torch.cuda.get_device_capability``, which
    trips a bare internal ``assert`` on an unknown index. A run configured with more devices than the host has (or with
    an explicit index that does not exist) must still fail in Lightning's own words, not with that ``AssertionError``.
    """

    @pytest.mark.parametrize(
        ("devices", "expected"),
        [
            pytest.param(4, [0, 1], id="count-above-device-count"),
            pytest.param([0, 3], [0], id="index-above-device-count"),
            pytest.param("0,3", [0], id="comma-index-above-device-count"),
            pytest.param("auto", [0, 1], id="auto-is-every-visible-device"),
        ],
    )
    def test_requested_indices_are_clamped_to_the_visible_devices(
        self, mocker, devices: int | str | list[int], expected: list[int]
    ) -> None:
        """Indices beyond ``torch.cuda.device_count()`` are dropped instead of forwarded to a per-device probe.

        The host here has two GPUs, so ``devices=4``, ``[0, 3]`` and ``"0,3"`` all name at least one device that does
        not exist — the shapes a user hits by copying a multi-GPU recipe onto a smaller machine.
        """
        mocker.patch("torch.cuda.device_count", return_value=2)

        assert _cuda_training_device_indices(devices) == expected

    def test_a_host_without_cuda_selects_no_device(self, mocker) -> None:
        """With no visible CUDA device every ``devices`` form resolves to an empty index list.

        This is the CPU-only case (CI, laptops): the callers then fall back to probing the current device, which reports
        no bfloat16 support because CUDA is unavailable.
        """
        mocker.patch("torch.cuda.device_count", return_value=0)

        assert _cuda_training_device_indices(2) == []


class TestCudaSupportsNativeBf16:
    """``_cuda_supports_native_bf16`` answers "native bfloat16 on this device?", unlike ``is_bf16_supported()``."""

    @pytest.mark.parametrize(
        ("capability", "hip", "native"),
        [
            pytest.param((7, 5), None, False, id="t4"),
            pytest.param((7, 0), None, False, id="v100"),
            pytest.param((8, 0), None, True, id="a100"),
            pytest.param((8, 9), None, True, id="rtx-4090"),
            pytest.param((7, 5), "5.7.31921", True, id="rocm-pre-ampere-shaped-capability"),
        ],
    )
    def test_native_means_compute_capability_8_or_newer(
        self, mocker, capability: tuple[int, int], hip: str | None, native: bool
    ) -> None:
        """Only Ampere and newer run bfloat16 natively on CUDA, even though ``is_bf16_supported()`` says True on a T4.

        Every ROCm device (``torch.version.hip`` truthy) is the exception: it short-circuits to native bf16 without
        consulting compute capability at all, so a pre-Ampere-shaped capability tuple paired with a truthy hip must
        still resolve to True, not fall through to the capability comparison.
        """
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.is_bf16_supported", return_value=True)
        mocker.patch("torch.cuda.get_device_capability", return_value=capability)
        mocker.patch("torch.version.hip", hip)

        assert _cuda_supports_native_bf16() is native

    def test_checks_the_requested_device_not_the_current_one(self, mocker) -> None:
        """On a host with a T4 at index 0 and an A100 at index 1, each index gets its own answer."""
        capabilities = {0: (7, 5), 1: (8, 0)}
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.get_device_capability", side_effect=lambda device=None: capabilities[device or 0])
        mocker.patch("torch.version.hip", None)

        assert _cuda_supports_native_bf16(0) is False, "the T4 at index 0 has no native bf16"
        assert _cuda_supports_native_bf16(1) is True, "the A100 at index 1 has native bf16"

    @pytest.mark.parametrize(
        "capability",
        [
            pytest.param((8, 0), id="a100-shaped-capability"),
            pytest.param((7, 5), id="t4-shaped-capability"),
        ],
    )
    def test_returns_false_when_cuda_is_unavailable_regardless_of_capability(
        self, mocker, capability: tuple[int, int]
    ) -> None:
        """``is_available() == False`` short-circuits to False before the capability check ever runs.

        The mocked capability is Ampere-shaped in one case, so a True result here could only come from skipping the
        ``is_available()`` guard, not from a capability read that never happens.
        """
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.cuda.get_device_capability", return_value=capability)
        mocker.patch("torch.version.hip", None)

        assert _cuda_supports_native_bf16() is False

    def test_accepts_a_torch_device_argument(self, mocker) -> None:
        """Type-hint completeness only: ``device`` accepts a ``torch.device``, matching its declared ``torch.device |
        int | None`` annotation.

        No production call site passes a ``torch.device`` here today -- ``auto_batch.py`` never calls
        ``_cuda_supports_native_bf16`` directly; it goes through ``_cuda_native_bf16_on_devices``, which only ever
        forwards int indices or ``None`` derived from ``TrainConfig.devices``. This case exists to cover the annotated
        type, not to reproduce a real call path.
        """
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.cuda.get_device_capability", return_value=(8, 0))
        mocker.patch("torch.version.hip", None)

        assert _cuda_supports_native_bf16(torch.device("cuda", 0)) is True


class TestCudaTrainingDeviceIndices:
    """``_cuda_training_device_indices`` turns a Lightning ``devices`` value into the CUDA indices it trains on."""

    @pytest.mark.parametrize("devices", ["auto", -1, 0, "gpu0"])
    def test_every_all_visible_sentinel_falls_through_to_every_device(self, mocker, devices: int | str) -> None:
        """``"auto"``, ``-1``, and ``0`` are the documented "every visible device" sentinels.

        A non-numeric typo like ``"gpu0"`` is not distinguished from those sentinels: it fails the same ``isdigit()``
        check ``"auto"`` does and silently takes the identical fallback path instead of raising. This test documents
        that identical behaviour explicitly rather than asserting a distinction that does not exist.
        """
        mocker.patch("torch.cuda.device_count", return_value=3)

        assert _cuda_training_device_indices(devices) == [0, 1, 2]

    def test_empty_sequence_returns_empty_not_every_device(self) -> None:
        """An empty explicit sequence (``devices=[]``) returns an empty list, not every visible device.

        The fallback to the current device (``[None]``) for an empty result happens one layer up, in
        ``_cuda_native_bf16_on_devices``, not inside this function.
        """
        assert _cuda_training_device_indices([]) == []


class TestCudaBf16SupportedOnDevices:
    """``_cuda_bf16_supported_on_devices`` gates an explicit ``amp_dtype="bf16"``, emulated bfloat16 included.

    The gate and the "this GPU only emulates bfloat16" warning beside it must read the same devices, so this asks
    ``torch.cuda.is_bf16_supported()``'s wider question for the GPUs a run trains on rather than the current device.
    """

    def test_a_device_that_only_emulates_bf16_still_counts_as_supported(self, mocker) -> None:
        """A pre-Ampere training GPU reports bfloat16 through emulation, so an explicit request is honoured.

        This is the T4/V100 case: ``amp_dtype="bf16"`` stays bf16 (with a warning about the cost) instead of silently
        falling back to fp16.
        """
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.version.hip", None)
        mocker.patch("torch.cuda.device_count", return_value=2)
        mocker.patch("torch.cuda.get_device_capability", return_value=(7, 5))
        mocker.patch("torch.cuda.is_bf16_supported", return_value=True)

        assert _cuda_bf16_supported_on_devices([1]) is True

    def test_no_bf16_even_through_emulation_is_unsupported(self, mocker) -> None:
        """When PyTorch reports no bfloat16 at all the gate is False, which is what triggers the fp16 fallback.

        A device below Ampere on a CUDA build without bfloat16 emulation, so neither half of the question passes.
        """
        mocker.patch("torch.cuda.is_available", return_value=True)
        mocker.patch("torch.version.hip", None)
        mocker.patch("torch.cuda.device_count", return_value=1)
        mocker.patch("torch.cuda.get_device_capability", return_value=(6, 1))
        mocker.patch("torch.cuda.is_bf16_supported", return_value=False)

        assert _cuda_bf16_supported_on_devices(1) is False

    def test_a_rocm_build_without_a_visible_gpu_is_unsupported(self, mocker) -> None:
        """No CUDA device means no bfloat16, even on a build whose bare PyTorch probe answers True.

        ``torch.cuda.is_bf16_supported()`` reads ``torch.version.hip`` before ``torch.cuda.is_available()``, so a ROCm
        wheel on a machine with no GPU returns True from it — the helper must not inherit that answer.
        """
        mocker.patch("torch.cuda.is_available", return_value=False)
        mocker.patch("torch.version.hip", "6.0.0")
        mocker.patch("torch.cuda.is_bf16_supported", return_value=True)

        assert _cuda_bf16_supported_on_devices(1) is False


class TestPretrainWeightsCompatibilityWarning:
    """Config-time warning for overrides that prevent pretrained weights from loading.

    These tests instantiate the variant *config* directly (not the wrapper class) so they do not touch the network, the
    cache, or any model construction.
    """

    def _capture(self, config_cls: type, **kwargs: object) -> list[warnings.WarningMessage]:
        """Instantiate ``config_cls(**kwargs)`` and return only the pretrain-compat warnings."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config_cls(**kwargs)
            return [w for w in caught if issubclass(w.category, PretrainWeightsCompatibilityWarning)]

    def test_default_construction_emits_no_warning(self) -> None:
        """Default variant construction must not warn — defaults match the published checkpoint."""
        assert self._capture(RFDETRNanoConfig) == []

    def test_encoder_registers_override_warns(self) -> None:
        """The dinov2-with-registers footgun: switching encoder away from the variant default."""
        captured = self._capture(RFDETRNanoConfig, encoder="dinov2_registers_windowed_small")
        assert len(captured) == 1
        message = str(captured[0].message)
        assert "encoder" in message
        assert "dinov2_registers_windowed_small" in message
        assert "dinov2_windowed_small" in message

    @pytest.mark.parametrize(
        "field, value",
        [
            ("hidden_dim", 384),
            ("dec_layers", 6),
            ("num_windows", 4),
            ("sa_nheads", 4),
            ("ca_nheads", 8),
            ("dec_n_points", 4),
            ("dim_feedforward", 1024),
            pytest.param("out_feature_indexes", [2, 5, 8, 11], id="out_feature_indexes"),
            pytest.param("projector_scale", ["P3", "P4"], id="projector_scale"),
            ("bbox_reparam", False),
            ("lite_refpoint_refine", False),
            ("layer_norm", False),
            ("two_stage", False),
            ("num_channels", 1),
        ],
    )
    def test_load_breaking_override_warns(self, field: str, value: object) -> None:
        """Each load-breaking architecture override fires the warning."""
        captured = self._capture(RFDETRNanoConfig, **{field: value})
        assert len(captured) == 1
        assert field in str(captured[0].message)

    @pytest.mark.parametrize(
        "config_cls, field, value",
        [
            pytest.param(RFDETRSegNanoConfig, "mask_downsample_ratio", 2, id="mask_downsample_ratio"),
            # patch_size already raises in load_pretrain_weights; this warning is defense-in-depth.
            # Value differs from RFDETRNanoConfig's default (16).
            pytest.param(RFDETRNanoConfig, "patch_size", 14, id="patch_size"),
            # RFDETRNanoConfig has segmentation_head=False; flipping it to True is the override,
            # which also raises at load time but the warning fires first.
            pytest.param(RFDETRNanoConfig, "segmentation_head", True, id="segmentation_head"),
        ],
    )
    def test_single_field_override_warns(self, config_cls: type, field: str, value: object) -> None:
        """A single breaking-field override on its variant config fires exactly one warning naming the field."""
        captured = self._capture(config_cls, **{field: value})
        assert len(captured) == 1
        assert field in str(captured[0].message)

    @pytest.mark.parametrize(
        "field, value",
        [
            ("num_queries", 200),
            ("num_queries", 300),
            ("group_detr", 8),
            ("num_classes", 5),
            ("resolution", 448),
            ("positional_encoding_size", 20),
        ],
    )
    def test_silent_field_overrides(self, field: str, value: object) -> None:
        """Fields that are auto-handled at load time must not emit a warning at config construction."""
        assert self._capture(RFDETRNanoConfig, **{field: value}) == []

    @pytest.mark.parametrize(
        "field, value",
        [
            ("num_queries", 400),
            ("group_detr", 20),
        ],
    )
    def test_increase_field_warns(self, field: str, value: object) -> None:
        """Increasing an integer field above the variant default warns — extra slots are randomly initialised."""
        captured = self._capture(RFDETRNanoConfig, **{field: value})
        assert len(captured) == 1
        assert field in str(captured[0].message)

    def test_pretrain_weights_none_warns(self) -> None:
        """Explicitly opting out of pretrained weights warns about training from scratch."""
        captured = self._capture(RFDETRNanoConfig, pretrain_weights=None)
        assert len(captured) == 1
        message = str(captured[0].message)
        assert "from scratch" in message
        assert "rf-detr-nano.pth" in message

    def test_pretrain_weights_none_only_one_warning(self) -> None:
        """When pretrain_weights=None, the architecture-overrides warning is suppressed.

        The from-scratch warning is the dominant message; we don't pile on with arch warnings.
        """
        captured = self._capture(
            RFDETRNanoConfig,
            pretrain_weights=None,
            encoder="dinov2_registers_windowed_small",
            hidden_dim=384,
        )
        assert len(captured) == 1
        assert "from scratch" in str(captured[0].message)

    def test_custom_pretrain_weights_path_suppresses_arch_warning(self) -> None:
        """Custom pretrain_weights path → defer to load-time detector — no config-time arch warning."""
        captured = self._capture(
            RFDETRNanoConfig,
            pretrain_weights="/tmp/my_custom.pth",
            encoder="dinov2_registers_windowed_small",
        )
        assert captured == []

    def test_multiple_overrides_consolidated_into_one_warning(self) -> None:
        """All overrides are listed in a single warning, not one warning per field."""
        captured = self._capture(
            RFDETRNanoConfig,
            encoder="dinov2_registers_windowed_small",
            hidden_dim=384,
            num_queries=400,
        )
        assert len(captured) == 1
        message = str(captured[0].message)
        for needle in ("encoder", "hidden_dim", "num_queries"):
            assert needle in message, f"expected {needle!r} in consolidated warning message"

    def test_warning_is_user_warning_subclass(self) -> None:
        """Confirms downstream filtering via UserWarning works."""
        assert issubclass(PretrainWeightsCompatibilityWarning, UserWarning)

    def test_modelconfig_with_required_fields_does_not_warn(self, sample_model_config: dict[str, object]) -> None:
        """Constructing the abstract ModelConfig with required fields cannot compare to defaults — no warning."""
        assert self._capture(ModelConfig, **sample_model_config) == []

    def test_breaking_field_with_default_factory_skips_comparison(self) -> None:
        """A subclass whose breaking field uses ``default_factory`` (so ``.default`` is ``PydanticUndefined``) must be
        silently skipped — we have nothing to compare against."""

        class _DefaultFactoryConfig(RFDETRNanoConfig):
            # default_factory leaves FieldInfo.default as PydanticUndefined while is_required() is False.
            encoder: str = Field(default_factory=lambda: "dinov2_windowed_small")

        assert self._capture(_DefaultFactoryConfig, encoder="dinov2_registers_windowed_small") == []

    def test_increase_field_when_required_skips_comparison(self) -> None:
        """A subclass where ``num_queries`` becomes required (no default) must be skipped."""

        class _RequiredNumQueriesConfig(RFDETRNanoConfig):
            num_queries: int  # type: ignore[misc]  # no default → required

        assert self._capture(_RequiredNumQueriesConfig, num_queries=400) == []

    def test_increase_field_with_non_int_default_skips_comparison(self) -> None:
        """A subclass where ``num_queries`` has a non-int default must be skipped (can't ``>`` compare)."""

        class _NonIntDefaultConfig(RFDETRNanoConfig):
            num_queries: Any = "300"  # type: ignore[assignment]  # non-int default

        assert self._capture(_NonIntDefaultConfig, num_queries="400") == []

    def test_explicit_variant_default_path_runs_arch_override_check(self) -> None:
        """Passing the variant's own published-default path string must still check arch overrides.

        Before the case-2 fix, any non-None explicit pretrain_weights bypassed the architecture-override check entirely
        — including when the user passed the exact variant default string such as "rf-detr-nano.pth".
        """
        captured = self._capture(
            RFDETRNanoConfig,
            pretrain_weights="rf-detr-nano.pth",
            encoder="dinov2_registers_windowed_small",
        )
        assert len(captured) == 1
        assert "encoder" in str(captured[0].message)

    def test_product_preserving_group_detr_increase_still_warns(self) -> None:
        """Increasing group_detr while halving num_queries still warns — check is per-field, not product-aware.

        This documents known current behaviour: the validator compares each field to its variant default independently,
        not the combined query-slot product.  A product- preserving change (group_detr=26, num_queries=150 vs defaults
        13, 300) warns for group_detr because 26 > 13, regardless of whether total slots are the same.
        """
        captured = self._capture(RFDETRNanoConfig, num_queries=150, group_detr=26)
        assert len(captured) == 1
        assert "group_detr" in str(captured[0].message)


class TestBreakingListIntegrity:
    """Guards against stale entries in the pretrain-compatibility breaking-field lists."""

    def test_all_breaking_fields_exist_in_model_config(self) -> None:
        """Every field guarded by the pretrain-compatibility check must exist in ModelConfig.model_fields.

        Catches typos and fields renamed/removed without updating the breaking lists.
        """
        all_breaking = {
            "encoder",
            "hidden_dim",
            "dec_layers",
            "num_windows",
            "sa_nheads",
            "ca_nheads",
            "dec_n_points",
            "dim_feedforward",
            "out_feature_indexes",
            "projector_scale",
            "bbox_reparam",
            "lite_refpoint_refine",
            "layer_norm",
            "two_stage",
            "patch_size",
            "segmentation_head",
            "num_channels",
            "num_queries",
            "group_detr",
        }
        stale = all_breaking - set(ModelConfig.model_fields.keys())
        assert not stale, f"Fields in breaking lists not in ModelConfig.model_fields: {stale}"


class TestTrainConfigAugmentationBackendSerialization:
    """Serialization contract for ``TrainConfig.augmentation_backend`` used by checkpoint writers (Item #6).

    ``BestModelCallback`` serializes the training config with a plain ``model_dump()`` before writing it into the
    ``.pth`` checkpoint's ``args``.  A ``@field_serializer`` renders the ``AugmentationBackend`` enum as its plain
    string value so the writer stays JSON-safe without a blanket ``model_dump(mode="json")`` that would silently coerce
    every *other* field's serialized shape (e.g. ``int`` loss coefficients to ``float``).
    """

    @pytest.fixture
    def dumped(self, tmp_path: Path) -> dict[str, Any]:
        """Plain ``model_dump()`` of a ``TrainConfig`` using the torchvision backend."""
        return TrainConfig(dataset_dir=str(tmp_path), augmentation_backend="torchvision").model_dump()

    def test_backend_dumps_to_json_safe_string(self, dumped: dict[str, Any]) -> None:
        """Plain ``model_dump()`` (as BestModelCallback uses) renders the enum as its ``str`` value."""
        assert dumped["augmentation_backend"] == AugmentationBackend.TV.value
        assert type(dumped["augmentation_backend"]) is str

    def test_dumped_backend_survives_json_sidecar(self, dumped: dict[str, Any]) -> None:
        """The dumped backend survives the ``json.dump(..., default=str)`` sidecar path unchanged."""
        round_tripped = json.loads(json.dumps(dumped, default=str))
        assert round_tripped["augmentation_backend"] == "torchvision"

    def test_dumped_backend_reconstructs_to_enum_member(self, dumped: dict[str, Any]) -> None:
        """Reloading a config from its dumped args restores the concrete ``AugmentationBackend`` member."""
        with warnings.catch_warnings():
            # A full dump sets deprecated fields; those warnings are unrelated to the backend round-trip under test.
            warnings.simplefilter("ignore", DeprecationWarning)
            reloaded = TrainConfig(**dumped)
        assert reloaded.augmentation_backend is AugmentationBackend.TV

    def test_plain_dump_does_not_coerce_int_field(self, dumped: dict[str, Any]) -> None:
        """Plain ``model_dump()`` keeps native field types — guards against a blanket ``mode="json"`` regression.

        Under ``model_dump(mode="json")`` this ``int`` default is silently coerced to ``float``; the
        ``field_serializer`` lets the writer stay on plain ``model_dump()`` so unrelated fields keep their shape.
        """
        assert type(dumped["keypoint_l1_loss_coef"]) is int

    @pytest.mark.parametrize(
        "sentinel",
        [
            "cpu",
            "auto",
        ],
    )
    def test_sentinel_backend_passes_through_as_string(self, tmp_path: Path, sentinel: str) -> None:
        """The ``"cpu"``/``"auto"`` auto-pick sentinels serialize unchanged as plain strings."""
        config = TrainConfig(dataset_dir=str(tmp_path), augmentation_backend=sentinel)
        dumped = config.model_dump()
        assert dumped["augmentation_backend"] == sentinel


class TestAugmentationBackendAvailability:
    """Availability probes use the backend-specific optional dependency contract."""

    @pytest.mark.parametrize(
        ("backend", "module_name", "importable"),
        [
            pytest.param(AugmentationBackend.ALBU, "albumentations", True, id="albumentations-present"),
            pytest.param(AugmentationBackend.KORNIA, "kornia.augmentation", True, id="kornia-present"),
            pytest.param(AugmentationBackend.GPU, "kornia.augmentation", True, id="gpu-alias-present"),
            pytest.param(AugmentationBackend.TV, "torchvision.transforms.v2", True, id="torchvision-v2-present"),
            pytest.param(AugmentationBackend.ALBU, "albumentations", False, id="albumentations-absent"),
            pytest.param(AugmentationBackend.KORNIA, "kornia.augmentation", False, id="kornia-absent"),
            pytest.param(AugmentationBackend.TV, "torchvision.transforms.v2", False, id="torchvision-v2-absent"),
        ],
    )
    def test_backend_availability_follows_its_required_module_probe(
        self, mocker, backend: AugmentationBackend, module_name: str, importable: bool
    ) -> None:
        """Each backend probes the module implementing its transform path and reports that probe's answer."""
        package_importable = mocker.patch.object(config_module, "_package_importable", return_value=importable)

        assert bool(backend._is_available()) is importable
        package_importable.assert_called_once_with(module_name)

    def test_gpu_is_an_alias_for_kornia(self) -> None:
        """The legacy GPU spelling retains Kornia availability semantics."""
        assert AugmentationBackend.GPU is AugmentationBackend.KORNIA


class TestTrainConfigAugmentationBackendConstruction:
    """Construction-time validation for ``TrainConfig.augmentation_backend`` (Item #7).

    The ``@field_validator(mode="before")`` only maps legacy alias strings (``"gpu"``, ``"tv"``, ``"albu"``) to their
    current form; it never runs custom logic for non-string input, so an already-concrete ``AugmentationBackend`` member
    must be accepted unchanged. Unrecognized strings must surface as a Pydantic ``ValidationError`` — not some other
    exception or a silent fallback to a default backend.
    """

    def test_unrecognized_backend_string_raises_validation_error(self, tmp_path: Path) -> None:
        """An unrecognized backend string raises ``ValidationError``, not a silent fallback."""
        with pytest.raises(ValidationError, match="augmentation_backend"):
            TrainConfig(dataset_dir=str(tmp_path), augmentation_backend="not_a_real_backend")

    def test_enum_member_accepted_directly_without_alias_lookup(self, tmp_path: Path) -> None:
        """Passing a concrete ``AugmentationBackend`` member at construction bypasses the alias map."""
        config = TrainConfig(dataset_dir=str(tmp_path), augmentation_backend=AugmentationBackend.ALBU)
        assert config.augmentation_backend is AugmentationBackend.ALBU

    @pytest.mark.parametrize(
        "alias,expected",
        [
            pytest.param("gpu", AugmentationBackend.KORNIA, id="gpu-aliases-to-kornia"),
            pytest.param("tv", AugmentationBackend.TV, id="tv-aliases-to-torchvision"),
            pytest.param("albu", AugmentationBackend.ALBU, id="albu-aliases-to-albumentations"),
        ],
    )
    def test_legacy_alias_string_resolves_to_current_enum_member(
        self, tmp_path: Path, alias: str, expected: AugmentationBackend
    ) -> None:
        """Legacy alias strings (``"gpu"``, ``"tv"``, ``"albu"``) still resolve to their current backend."""
        config = TrainConfig(dataset_dir=str(tmp_path), augmentation_backend=alias)
        assert config.augmentation_backend is expected
