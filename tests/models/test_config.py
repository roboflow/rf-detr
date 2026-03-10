# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest
from pydantic import ValidationError

from rfdetr.config import (
    ModelConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
    SegmentationTrainConfig,
    TrainConfig,
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


class TestSegmentationTrainConfigNumSelect:
    """Unit tests for SegmentationTrainConfig.num_select default and per-model values."""

    def test_defaults_to_none(self) -> None:
        config = SegmentationTrainConfig(dataset_dir="/tmp")
        assert config.num_select is None

    def test_explicit_value_is_accepted(self) -> None:
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
            pytest.param("ema_update_interval", 0, id="ema_update_interval_zero"),
            pytest.param("prefetch_factor", 0, id="prefetch_factor_zero"),
        ],
    )
    def test_interval_and_prefetch_reject_non_positive_values(self, tmp_path, field, value):
        """eval/EMA intervals and prefetch_factor must be >= 1 when provided."""
        with pytest.raises((ValueError, ValidationError)):
            self._tc(tmp_path, **{field: value})


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
