# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import warnings
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError

from rfdetr.config import (
    ModelConfig,
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
    _detect_device,
)


@pytest.fixture
def sample_model_config() -> dict[str, object]:
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


class TestModelConfigValidation:
    def test_rejects_unknown_fields(self, sample_model_config) -> None:
        sample_model_config["unknown"] = "value"

        with pytest.raises(ValidationError, match=r"Unknown parameter\(s\): 'unknown'"):
            ModelConfig(**sample_model_config)

    def test_rejects_unknown_attribute_assignment(self, sample_model_config) -> None:
        config = ModelConfig(**sample_model_config)

        with pytest.raises(ValueError, match=r"Unknown attribute: 'unknown'\."):
            setattr(config, "unknown", "value")

    def test_accepts_indexed_cuda_device_string(self, sample_model_config) -> None:
        config = ModelConfig(**sample_model_config, device="cuda:1")
        assert config.device == "cuda:1"

    def test_accepts_torch_device(self, sample_model_config) -> None:
        config = ModelConfig(**sample_model_config, device=torch.device("cuda:2"))
        assert config.device == "cuda:2"

    def test_rejects_non_string_non_torch_device_with_validation_error(self, sample_model_config) -> None:
        with pytest.raises(ValidationError, match="device must be a string or torch\\.device\\."):
            ModelConfig(**sample_model_config, device=123)

    def test_rejects_invalid_device_string(self, sample_model_config) -> None:
        with pytest.raises(ValidationError, match="Invalid device specifier: 'notadevice'\\."):
            ModelConfig(**sample_model_config, device="notadevice")

    @pytest.mark.parametrize(
        "encoder",
        [
            pytest.param("dinov2_windowed_small", id="windowed_small"),
            pytest.param("dinov2_windowed_base", id="windowed_base"),
            pytest.param("dinov2_registers_windowed_small", id="registers_windowed_small"),
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


class TestSegmentationTrainConfigNumSelect:
    """Unit tests for SegmentationTrainConfig.num_select default and per-model values."""

    def test_defaults_to_none(self) -> None:
        config = SegmentationTrainConfig(dataset_dir="/tmp")
        assert config.num_select is None

    def test_explicit_value_is_accepted(self) -> None:
        # Explicitly setting num_select on SegmentationTrainConfig is deprecated (Item #3).
        with pytest.warns(DeprecationWarning, match="TrainConfig.num_select is deprecated"):
            config = SegmentationTrainConfig(dataset_dir="/tmp", num_select=42)
        assert config.num_select == 42

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
        assert config_class().num_select == expected_num_select


class TestTrainConfigT42PromotedFields:
    """T4-2: Promoted fields exist with correct defaults; device field is absent."""

    def _tc(self, tmp_path, **kwargs):
        defaults = dict(dataset_dir=str(tmp_path), output_dir=str(tmp_path), tensorboard=False)
        defaults.update(kwargs)
        return TrainConfig(**defaults)

    # --- device field removed ---

    def test_device_not_in_model_fields(self):
        """device must not appear in TrainConfig.model_fields (PTL auto-detects accelerator)."""
        assert "device" not in TrainConfig.model_fields

    def test_device_kwarg_silently_ignored(self, tmp_path):
        """Passing device= to TrainConfig is silently ignored (extra='ignore'); PTL absorbs it."""
        # TrainConfig uses Pydantic default extra='ignore', so unknown kwargs don't raise.
        tc = self._tc(tmp_path, device="cpu")
        assert not hasattr(tc, "device")  # field not set on the instance

    # --- promoted fields: defaults ---

    def test_clip_max_norm_default(self, tmp_path):
        """clip_max_norm defaults to 0.1."""
        assert self._tc(tmp_path).clip_max_norm == pytest.approx(0.1)

    def test_seed_default_is_none(self, tmp_path):
        """seed defaults to None (no seeding)."""
        assert self._tc(tmp_path).seed is None

    def test_sync_bn_default_is_false(self, tmp_path):
        """sync_bn defaults to False."""
        assert self._tc(tmp_path).sync_bn is False

    def test_fp16_eval_default_is_false(self, tmp_path):
        """fp16_eval defaults to False."""
        assert self._tc(tmp_path).fp16_eval is False

    def test_lr_scheduler_default_is_step(self, tmp_path):
        """lr_scheduler defaults to 'step'."""
        assert self._tc(tmp_path).lr_scheduler == "step"

    def test_lr_min_factor_default(self, tmp_path):
        """lr_min_factor defaults to 0.0."""
        assert self._tc(tmp_path).lr_min_factor == pytest.approx(0.0)

    def test_dont_save_weights_default_is_false(self, tmp_path):
        """dont_save_weights defaults to False."""
        assert self._tc(tmp_path).dont_save_weights is False

    def test_run_test_default_is_false(self, tmp_path):
        """run_test defaults to False to avoid extra full-dataset test passes."""
        assert self._tc(tmp_path).run_test is False

    def test_eval_interval_default_is_one(self, tmp_path):
        """eval_interval defaults to 1 (evaluate each epoch)."""
        assert self._tc(tmp_path).eval_interval == 1

    def test_skip_best_epochs_default_is_zero(self, tmp_path):
        """skip_best_epochs defaults to 0 for backward compatibility."""
        assert self._tc(tmp_path).skip_best_epochs == 0

    def test_ema_update_interval_default_is_one(self, tmp_path):
        """ema_update_interval defaults to 1 (update every step)."""
        assert self._tc(tmp_path).ema_update_interval == 1

    def test_compute_val_loss_default_is_true(self, tmp_path):
        """compute_val_loss defaults to True."""
        assert self._tc(tmp_path).compute_val_loss is True

    def test_compute_test_loss_default_is_true(self, tmp_path):
        """compute_test_loss defaults to True."""
        assert self._tc(tmp_path).compute_test_loss is True

    # --- promoted fields: accept explicit values ---

    @pytest.mark.parametrize(
        "field, value",
        [
            pytest.param("clip_max_norm", 0.5, id="clip_max_norm"),
            pytest.param("seed", 42, id="seed"),
            pytest.param("sync_bn", True, id="sync_bn"),
            pytest.param("fp16_eval", True, id="fp16_eval"),
            pytest.param("lr_scheduler", "cosine", id="lr_scheduler_cosine"),
            pytest.param("lr_min_factor", 0.01, id="lr_min_factor"),
            pytest.param("dont_save_weights", True, id="dont_save_weights"),
            pytest.param("run_test", True, id="run_test"),
            pytest.param("eval_interval", 3, id="eval_interval"),
            pytest.param("skip_best_epochs", 3, id="skip_best_epochs"),
            pytest.param("ema_update_interval", 4, id="ema_update_interval"),
            pytest.param("compute_val_loss", False, id="compute_val_loss"),
            pytest.param("compute_test_loss", False, id="compute_test_loss"),
            pytest.param("train_log_sync_dist", True, id="train_log_sync_dist"),
            pytest.param("train_log_on_step", True, id="train_log_on_step"),
            pytest.param("log_per_class_metrics", False, id="log_per_class_metrics"),
            pytest.param("prefetch_factor", 4, id="prefetch_factor"),
            pytest.param("pin_memory", False, id="pin_memory"),
            pytest.param("persistent_workers", False, id="persistent_workers"),
        ],
    )
    def test_promoted_field_accepts_explicit_value(self, tmp_path, field, value):
        """Each promoted field accepts an explicit value."""
        tc = self._tc(tmp_path, **{field: value})
        assert getattr(tc, field) == value

    def test_lr_scheduler_rejects_invalid_value(self, tmp_path):
        """lr_scheduler must reject values other than 'step' and 'cosine'."""
        with pytest.raises((ValueError, ValidationError)):
            self._tc(tmp_path, lr_scheduler="cyclic")

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            pytest.param("eval_interval", 0, id="eval_interval_zero"),
            pytest.param("skip_best_epochs", -1, id="skip_best_epochs_negative"),
            pytest.param("ema_update_interval", 0, id="ema_update_interval_zero"),
            pytest.param("prefetch_factor", 0, id="prefetch_factor_zero"),
        ],
    )
    def test_interval_and_prefetch_reject_non_positive_values(self, tmp_path, field, value):
        """eval/EMA intervals and prefetch_factor must be >= 1 when provided."""
        with pytest.raises((ValueError, ValidationError)):
            self._tc(tmp_path, **{field: value})

    def test_batch_size_auto_is_accepted(self, tmp_path):
        """batch_size accepts the special 'auto' value."""
        tc = self._tc(tmp_path, batch_size="auto")
        assert tc.batch_size == "auto"

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
            self._tc(tmp_path, **{field: value})

    @pytest.mark.parametrize("ema_headroom", [0.0, 1.5])
    def test_auto_batch_ema_headroom_must_be_in_open_one(self, tmp_path, ema_headroom):
        """auto_batch_ema_headroom must be in (0, 1]."""
        with pytest.raises((ValueError, ValidationError)):
            self._tc(tmp_path, auto_batch_ema_headroom=ema_headroom)


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

    def _mc(self, **kwargs):
        from rfdetr.config import RFDETRBaseConfig

        defaults = dict(pretrain_weights=None, device="cpu", num_classes=3)
        defaults.update(kwargs)
        return RFDETRBaseConfig(**defaults)

    def test_clip_max_norm_forwarded_to_trainer(self, tmp_path):
        """gradient_clip_val on the Trainer matches TrainConfig.clip_max_norm."""
        from rfdetr.training import build_trainer

        trainer = build_trainer(self._tc(tmp_path, clip_max_norm=0.25), self._mc())
        assert trainer.gradient_clip_val == pytest.approx(0.25)

    def test_seed_not_applied_in_build_trainer_factory(self, tmp_path):
        """Seeding is deferred to RFDETRModule.on_fit_start, not build_trainer()."""
        import unittest.mock as mock

        from rfdetr.training import build_trainer

        with mock.patch("pytorch_lightning.seed_everything") as mock_seed:
            build_trainer(self._tc(tmp_path, seed=99), self._mc())
        mock_seed.assert_not_called()

    def test_sync_bn_forwarded_to_trainer(self, tmp_path):
        """sync_batchnorm=True is passed to Trainer when TrainConfig.sync_bn is True."""
        import unittest.mock as mock

        from rfdetr.training import build_trainer

        captured_kwargs = {}

        real_trainer_init = __import__("pytorch_lightning").Trainer.__init__

        def _capture_init(self_t, **kwargs):
            captured_kwargs.update(kwargs)
            real_trainer_init(self_t, **kwargs)

        with mock.patch("rfdetr.training.trainer.Trainer.__init__", _capture_init):
            build_trainer(self._tc(tmp_path, sync_bn=True), self._mc())

        assert captured_kwargs.get("sync_batchnorm") is True


class TestDeprecatedTrainConfigFields:
    """Item #3 Phase A: TrainConfig fields deprecated in favour of ModelConfig ownership."""

    def _tc(self, **kwargs):
        defaults = dict(dataset_dir="/tmp")
        defaults.update(kwargs)
        return TrainConfig(**defaults)

    @pytest.mark.parametrize(
        "field,value",
        [
            pytest.param("group_detr", 5, id="group_detr"),
            pytest.param("ia_bce_loss", False, id="ia_bce_loss"),
            pytest.param("segmentation_head", True, id="segmentation_head"),
            pytest.param("num_select", 100, id="num_select"),
        ],
    )
    def test_explicitly_set_deprecated_field_emits_warning(self, field, value) -> None:
        """Setting a deprecated TrainConfig field explicitly must emit DeprecationWarning."""
        with pytest.warns(DeprecationWarning, match=f"TrainConfig\\.{field} is deprecated"):
            self._tc(**{field: value})

    def test_default_group_detr_no_warning(self, recwarn) -> None:
        """TrainConfig() without explicit group_detr must NOT warn."""
        self._tc()
        depr_warnings = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
        assert not depr_warnings, f"Unexpected DeprecationWarning: {depr_warnings}"

    def test_segmentation_train_config_no_warning_on_default_fields(self, recwarn) -> None:
        """SegmentationTrainConfig() must NOT warn for its class-level defaults.

        segmentation_head=True and num_select=None are SegmentationTrainConfig defaults,
        not explicitly set by the user — they must not trigger DeprecationWarning.
        """
        SegmentationTrainConfig(dataset_dir="/tmp")
        depr_warnings = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
        assert not depr_warnings, f"Unexpected DeprecationWarning: {depr_warnings}"


class TestDeprecatedModelConfigClsLossCoef:
    """Item #3 Phase A: ModelConfig.cls_loss_coef deprecated in favour of TrainConfig ownership."""

    def test_explicit_cls_loss_coef_emits_warning(self) -> None:
        """Setting cls_loss_coef on ModelConfig explicitly must emit DeprecationWarning."""
        sample = dict(
            encoder="dinov2_windowed_small",
            out_feature_indexes=[1, 2, 3],
            dec_layers=3,
            projector_scale=["P3"],
            hidden_dim=256,
            patch_size=14,
            num_windows=2,
            sa_nheads=8,
            ca_nheads=8,
            dec_n_points=4,
            resolution=384,
            positional_encoding_size=256,
        )
        with pytest.warns(DeprecationWarning, match="ModelConfig\\.cls_loss_coef is deprecated"):
            ModelConfig(**sample, cls_loss_coef=2.0)

    def test_default_cls_loss_coef_no_warning(self, recwarn) -> None:
        """RFDETRBaseConfig() without explicit cls_loss_coef must NOT warn."""
        RFDETRBaseConfig(pretrain_weights=None, device="cpu")
        depr_warnings = [w for w in recwarn.list if issubclass(w.category, DeprecationWarning)]
        assert not depr_warnings, f"Unexpected DeprecationWarning: {depr_warnings}"


class TestSyncPEWithResolutionAtConstruction:
    """Tests for the _sync_pe_with_resolution model_validator.

    When a user provides a custom resolution at construction time (e.g.,
    ``RFDETRLarge(resolution=640)``), positional_encoding_size must be updated
    proportionally for configs where the default PE is formula-derived
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

    @patch("rfdetr.config.torch")
    def test_falls_back_to_cuda_when_accelerator_module_absent(self, mock_torch: MagicMock) -> None:
        """Returns 'cuda' via legacy fallback when torch.accelerator lacks current_accelerator (PyTorch < 2.4)."""
        mock_torch.accelerator = MagicMock(spec=[])  # no current_accelerator → hasattr returns False → fallback
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = False
        assert _detect_device() == "cuda"

    @patch("rfdetr.config.torch")
    def test_returns_cpu_when_current_accelerator_raises(self, mock_torch: MagicMock) -> None:
        """Returns 'cpu' directly from the except handler when current_accelerator() raises RuntimeError."""
        mock_torch.accelerator.current_accelerator.side_effect = RuntimeError("no device")
        assert _detect_device() == "cpu"

    @patch("rfdetr.config.torch")
    def test_returns_cpu_when_no_gpu_available(self, mock_torch: MagicMock) -> None:
        """Returns 'cpu' when accelerator is absent and neither CUDA nor MPS is available."""
        mock_torch.accelerator = MagicMock(spec=[])  # no current_accelerator → fallback branch
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        assert _detect_device() == "cpu"


class TestPretrainWeightsCompatibilityWarning:
    """Config-time warning for overrides that prevent pretrained weights from loading.

    These tests instantiate the variant *config* directly (not the wrapper class)
    so they do not touch the network, the cache, or any model construction.
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
            pytest.param("hidden_dim", 384, id="hidden_dim"),
            pytest.param("dec_layers", 6, id="dec_layers"),
            pytest.param("num_windows", 4, id="num_windows"),
            pytest.param("sa_nheads", 4, id="sa_nheads"),
            pytest.param("ca_nheads", 8, id="ca_nheads"),
            pytest.param("dec_n_points", 4, id="dec_n_points"),
            pytest.param("out_feature_indexes", [2, 5, 8, 11], id="out_feature_indexes"),
            pytest.param("projector_scale", ["P3", "P4"], id="projector_scale"),
            pytest.param("bbox_reparam", False, id="bbox_reparam"),
            pytest.param("lite_refpoint_refine", False, id="lite_refpoint_refine"),
            pytest.param("layer_norm", False, id="layer_norm"),
            pytest.param("two_stage", False, id="two_stage"),
            pytest.param("num_channels", 1, id="num_channels"),
        ],
    )
    def test_load_breaking_override_warns(self, field: str, value: object) -> None:
        """Each load-breaking architecture override fires the warning."""
        captured = self._capture(RFDETRNanoConfig, **{field: value})
        assert len(captured) == 1
        assert field in str(captured[0].message)

    def test_mask_downsample_ratio_warns_on_seg_variant(self) -> None:
        """``mask_downsample_ratio`` change is silently miscalibrating; must warn at config time."""
        captured = self._capture(RFDETRSegNanoConfig, mask_downsample_ratio=2)
        assert len(captured) == 1
        assert "mask_downsample_ratio" in str(captured[0].message)

    def test_patch_size_override_warns_defense_in_depth(self) -> None:
        """patch_size already raises in load_pretrain_weights; the new warning is defense-in-depth.

        We change patch_size to a value that differs from RFDETRNanoConfig's default (16).
        """
        captured = self._capture(RFDETRNanoConfig, patch_size=14)
        assert len(captured) == 1
        assert "patch_size" in str(captured[0].message)

    def test_segmentation_head_override_warns(self) -> None:
        """segmentation_head also raises at load time but warning fires first."""
        # RFDETRNanoConfig has segmentation_head=False; flipping it to True is the override.
        captured = self._capture(RFDETRNanoConfig, segmentation_head=True)
        assert len(captured) == 1
        assert "segmentation_head" in str(captured[0].message)

    @pytest.mark.parametrize(
        "field, value",
        [
            pytest.param("num_queries", 200, id="num_queries_decrease"),
            pytest.param("num_queries", 300, id="num_queries_equal"),
            pytest.param("group_detr", 8, id="group_detr_decrease"),
            pytest.param("num_classes", 5, id="num_classes"),
            pytest.param("resolution", 448, id="resolution"),
            pytest.param("positional_encoding_size", 20, id="positional_encoding_size"),
        ],
    )
    def test_silent_field_overrides(self, field: str, value: object) -> None:
        """Fields that are auto-handled at load time must not emit a warning at config construction."""
        assert self._capture(RFDETRNanoConfig, **{field: value}) == []

    @pytest.mark.parametrize(
        "field, value",
        [
            pytest.param("num_queries", 400, id="num_queries"),
            pytest.param("group_detr", 20, id="group_detr"),
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
        """A subclass whose breaking field uses ``default_factory`` (so ``.default`` is
        ``PydanticUndefined``) must be silently skipped — we have nothing to compare against.
        """
        from pydantic import Field

        class _DefaultFactoryConfig(RFDETRNanoConfig):
            # Field uses default_factory → FieldInfo.default is PydanticUndefined,
            # but is_required() is False.  Hits the `continue` on the
            # PydanticUndefined check.
            encoder: str = Field(default_factory=lambda: "dinov2_windowed_small")

        assert self._capture(_DefaultFactoryConfig, encoder="dinov2_registers_windowed_small") == []

    def test_increase_field_when_required_skips_comparison(self) -> None:
        """A subclass where ``num_queries`` becomes required (no default) must be skipped."""

        class _RequiredNumQueriesConfig(RFDETRNanoConfig):
            num_queries: int  # type: ignore[misc]  # no default → required

        assert self._capture(_RequiredNumQueriesConfig, num_queries=400) == []

    def test_increase_field_with_non_int_default_skips_comparison(self) -> None:
        """A subclass where ``num_queries`` has a non-int default must be skipped (can't ``>`` compare)."""
        from typing import Any

        class _NonIntDefaultConfig(RFDETRNanoConfig):
            num_queries: Any = "300"  # type: ignore[assignment]  # non-int default

        assert self._capture(_NonIntDefaultConfig, num_queries="400") == []

    def test_explicit_variant_default_path_runs_arch_override_check(self) -> None:
        """Passing the variant's own published-default path string must still check arch overrides.

        Before the case-2 fix, any non-None explicit pretrain_weights bypassed the
        architecture-override check entirely — including when the user passed the exact
        variant default string such as "rf-detr-nano.pth".
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

        This documents known current behaviour: the validator compares each field to its
        variant default independently, not the combined query-slot product.  A product-
        preserving change (group_detr=26, num_queries=150 vs defaults 13, 300) warns for
        group_detr because 26 > 13, regardless of whether total slots are the same.
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
