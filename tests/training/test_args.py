# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for build_namespace — the canonical config-to-Namespace mapping."""

import pytest

from rfdetr._namespace import build_namespace


class TestBuildNamespace:
    """build_namespace maps ModelConfig + TrainConfig to a Namespace."""

    def test_returns_namespace(self, base_model_config, base_train_config):
        """The return value has attribute access (argparse.Namespace or similar)."""
        args = build_namespace(base_model_config(), base_train_config())
        assert hasattr(args, "encoder")

    def test_forwards_model_config_fields(self, base_model_config, base_train_config):
        """All key ModelConfig fields are faithfully mapped."""
        mc = base_model_config(num_classes=7)
        args = build_namespace(mc, base_train_config())

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
        args = build_namespace(base_model_config(), tc)

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

    def test_forwards_promoted_train_fields(self, base_model_config, base_train_config):
        """Promoted TrainConfig fields are forwarded to the legacy namespace."""
        tc = base_train_config(clip_max_norm=0.35, seed=123, sync_bn=True, fp16_eval=True)
        args = build_namespace(base_model_config(), tc)

        assert args.clip_max_norm == pytest.approx(0.35)
        assert args.seed == 123
        assert args.sync_bn is True
        assert args.fp16_eval is True

    def test_seed_falls_back_to_legacy_default_when_unset(self, base_model_config, base_train_config):
        """seed defaults to 42 in the legacy namespace when TrainConfig.seed is None."""
        tc = base_train_config(seed=None)
        args = build_namespace(base_model_config(), tc)
        assert args.seed == 42

    def test_forwards_dataset_fields(self, base_model_config, base_train_config):
        """Dataset-routing fields are forwarded to the Namespace."""
        tc = base_train_config(multi_scale=True, expanded_scales=True, dataset_file="coco")
        args = build_namespace(base_model_config(), tc)

        assert args.multi_scale is True
        assert args.expanded_scales is True
        assert args.dataset_file == "coco"

    def test_num_queries_from_subclass_config(self, base_model_config, base_train_config):
        """num_queries is read from subclass config attributes."""
        mc = base_model_config()  # RFDETRBaseConfig has num_queries=300
        args = build_namespace(mc, base_train_config())
        assert args.num_queries == 300

    def test_resume_none_becomes_empty_string(self, base_model_config, base_train_config):
        """resume=None (the default) is converted to '' for the legacy Namespace."""
        tc = base_train_config()
        assert tc.resume is None
        args = build_namespace(base_model_config(), tc)
        assert args.resume == ""

    def test_segmentation_extras_forwarded_from_seg_config(self, base_model_config, seg_train_config):
        """SegmentationTrainConfig mask loss coefficients are forwarded."""
        mc = base_model_config(segmentation_head=True)
        tc = seg_train_config()
        args = build_namespace(mc, tc)

        assert args.mask_ce_loss_coef == pytest.approx(5.0)
        assert args.mask_dice_loss_coef == pytest.approx(5.0)

    def test_segmentation_extras_default_for_plain_config(self, base_model_config, base_train_config):
        """mask_* attributes default to 5.0 for a plain TrainConfig (not segmentation)."""
        args = build_namespace(base_model_config(), base_train_config())
        assert args.mask_ce_loss_coef == pytest.approx(5.0)
        assert args.mask_dice_loss_coef == pytest.approx(5.0)

    def test_segmentation_head_flag_forwarded(self, base_model_config, base_train_config):
        """segmentation_head=True from ModelConfig reaches the Namespace."""
        mc = base_model_config(segmentation_head=True)
        args = build_namespace(mc, base_train_config())
        assert args.segmentation_head is True

    def test_no_deprecation_warning_emitted(self, base_model_config, base_train_config):
        """build_namespace must not emit any DeprecationWarning or FutureWarning."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            build_namespace(base_model_config(), base_train_config())

        bad = [w for w in caught if issubclass(w.category, (DeprecationWarning, FutureWarning))]
        assert not bad
