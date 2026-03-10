# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Package-level pytest fixtures for tests/training/.

Provides cross-test cleanup that prevents class-level state from leaking
between individual tests in the training/ test package, plus shared config
factory fixtures used across multiple test modules.
"""

import pytest

from rfdetr.config import RFDETRBaseConfig, SegmentationTrainConfig, TrainConfig

# ---------------------------------------------------------------------------
# Shared config factory fixtures (used by test_module, test_datamodule,
# test_args — avoids duplicate fixture definitions across files)
# ---------------------------------------------------------------------------


@pytest.fixture
def base_model_config():
    """Factory fixture — call with **overrides to get a minimal RFDETRBaseConfig."""

    def _make(**overrides):
        defaults = dict(pretrain_weights=None, device="cpu", num_classes=5)
        defaults.update(overrides)
        return RFDETRBaseConfig(**defaults)

    return _make


@pytest.fixture
def base_train_config(tmp_path):
    """Factory fixture — call with **overrides to get a minimal TrainConfig.

    tmp_path is injected automatically so test methods do not need to declare it.
    """

    def _make(**overrides):
        defaults = dict(
            dataset_dir=str(tmp_path / "dataset"),
            output_dir=str(tmp_path / "output"),
            epochs=10,
            lr=1e-4,
            lr_encoder=1.5e-4,
            batch_size=2,
            weight_decay=1e-4,
            lr_drop=8,
            warmup_epochs=1.0,
            drop_path=0.0,
            multi_scale=False,
            expanded_scales=False,
            do_random_resize_via_padding=False,
            grad_accum_steps=1,
            num_workers=0,
            tensorboard=False,
        )
        defaults.update(overrides)
        return TrainConfig(**defaults)

    return _make


@pytest.fixture
def seg_train_config(tmp_path):
    """Factory fixture — call with **overrides to get a minimal SegmentationTrainConfig.

    tmp_path is injected automatically so test methods do not need to declare it.
    """

    def _make(**overrides):
        defaults = dict(
            dataset_dir=str(tmp_path / "dataset"),
            output_dir=str(tmp_path / "output"),
            epochs=10,
            batch_size=2,
            multi_scale=False,
            expanded_scales=False,
            do_random_resize_via_padding=False,
            grad_accum_steps=1,
            drop_path=0.0,
            num_workers=0,
            tensorboard=False,
        )
        defaults.update(overrides)
        return SegmentationTrainConfig(**defaults)

    return _make


# ---------------------------------------------------------------------------
# Class-level isolation
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_rfdetr_module_trainer_property():
    """Restore RFDETRModule.trainer to the LightningModule parent property after each test.

    Several unit tests in test_module.py patch the ``trainer`` property directly
    on the ``RFDETRModule`` class (``type(module).trainer = property(...)``).
    Without cleanup this mutates the class for the remainder of the session and
    breaks ``Trainer.fit()`` calls in smoke tests (PTL cannot set ``.trainer``
    on the module because the patched property has no setter).

    This fixture deletes any class-level override from ``RFDETRModule.__dict__``
    after every test, so the next test starts with a clean class that inherits
    PTL's read/write ``trainer`` descriptor from ``LightningModule``.
    """
    yield
    # Lazy import so the fixture does not force module import at collection time.
    from rfdetr.training.module import RFDETRModule

    if "trainer" in RFDETRModule.__dict__:
        delattr(RFDETRModule, "trainer")
