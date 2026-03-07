# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for _build_args_from_configs — the canonical config-to-Namespace mapping."""

import warnings

import pytest

from rfdetr.lit._args import _build_args_from_configs


class TestBuildArgsFromConfigs:
    """_build_args_from_configs maps ModelConfig + TrainConfig to a Namespace."""

    def test_returns_namespace(self, base_model_config, base_train_config):
        """The return value has attribute access (argparse.Namespace or similar)."""
        args = _build_args_from_configs(base_model_config(), base_train_config())
        assert hasattr(args, "encoder")

    def test_forwards_model_config_fields(self, base_model_config, base_train_config):
        """All key ModelConfig fields are faithfully mapped."""
        mc = base_model_config(num_classes=7)
        args = _build_args_from_configs(mc, base_train_config())

        assert args.encoder == mc.encoder
        assert args.num_classes == 7
        assert args.hidden_dim == mc.hidden_dim
        assert args.resolution == mc.resolution
        assert args.patch_size == mc.patch_size
        assert args.num_windows == mc.num_windows
        assert args.segmentation_head == mc.segmentation_head
        assert args.positional_encoding_size == mc.positional_encoding_size

    def test_forwards_train_config_fields(self, base_model_config, base_train_config):
        """All key TrainConfig fields are faithfully mapped."""
        tc = base_train_config(
            lr=3e-4,
            epochs=20,
            weight_decay=5e-5,
            batch_size=4,
            num_workers=0,
            eval_interval=3,
            log_per_class_metrics=False,
            train_log_sync_dist=True,
            train_log_on_step=True,
            compute_val_loss=False,
            compute_test_loss=False,
            ema_update_interval=2,
            prefetch_factor=4,
        )
        args = _build_args_from_configs(base_model_config(), tc)

        assert args.lr == pytest.approx(3e-4)
        assert args.epochs == 20
        assert args.weight_decay == pytest.approx(5e-5)
        assert args.batch_size == 4
        assert args.num_workers == 0
        assert args.eval_interval == 3
        assert args.log_per_class_metrics is False
        assert args.train_log_sync_dist is True
        assert args.train_log_on_step is True
        assert args.compute_val_loss is False
        assert args.compute_test_loss is False
        assert args.ema_update_interval == 2
        assert args.prefetch_factor == 4

    def test_forwards_dataset_fields(self, base_model_config, base_train_config):
        """Dataset-routing fields are forwarded to the Namespace."""
        tc = base_train_config(multi_scale=True, expanded_scales=True, dataset_file="coco")
        args = _build_args_from_configs(base_model_config(), tc)

        assert args.multi_scale is True
        assert args.expanded_scales is True
        assert args.dataset_file == "coco"

    def test_num_queries_from_subclass_config(self, base_model_config, base_train_config):
        """num_queries is read from subclass config attributes."""
        mc = base_model_config()  # RFDETRBaseConfig has num_queries=300
        args = _build_args_from_configs(mc, base_train_config())
        assert args.num_queries == 300

    def test_resume_none_becomes_empty_string(self, base_model_config, base_train_config):
        """resume=None (the default) is converted to '' for the legacy Namespace."""
        tc = base_train_config()
        assert tc.resume is None
        args = _build_args_from_configs(base_model_config(), tc)
        assert args.resume == ""

    def test_segmentation_extras_forwarded_from_seg_config(self, base_model_config, seg_train_config):
        """SegmentationTrainConfig mask loss coefficients are forwarded."""
        mc = base_model_config(segmentation_head=True)
        tc = seg_train_config()
        args = _build_args_from_configs(mc, tc)

        assert args.mask_ce_loss_coef == pytest.approx(5.0)
        assert args.mask_dice_loss_coef == pytest.approx(5.0)

    def test_segmentation_extras_default_for_plain_config(self, base_model_config, base_train_config):
        """mask_* attributes default to 5.0 for a plain TrainConfig (not segmentation)."""
        args = _build_args_from_configs(base_model_config(), base_train_config())
        assert args.mask_ce_loss_coef == pytest.approx(5.0)
        assert args.mask_dice_loss_coef == pytest.approx(5.0)

    def test_segmentation_head_flag_forwarded(self, base_model_config, base_train_config):
        """segmentation_head=True from ModelConfig reaches the Namespace."""
        mc = base_model_config(segmentation_head=True)
        args = _build_args_from_configs(mc, base_train_config())
        assert args.segmentation_head is True

    def test_internal_populate_args_deprecation_warning_is_suppressed(self, base_model_config, base_train_config):
        """Internal shim call must not leak populate_args FutureWarning to users."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _build_args_from_configs(base_model_config(), base_train_config())

        leaked = [
            warning
            for warning in caught
            if issubclass(warning.category, FutureWarning) and "populate_args" in str(warning.message)
        ]
        assert not leaked
